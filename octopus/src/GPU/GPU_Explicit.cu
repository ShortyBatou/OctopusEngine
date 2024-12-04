#include "GPU/GPU_Explicit.h"
#include "GPU/GPU_FEM_Material.h"
#include <GPU/CUMatrix.h>


__global__ void kernel_solve(
    // nb_thread, nb quadrature per elements, nb vertices in element
    const int n, const int nb_quadrature, const int elem_nb_verts,
    const Material material, const scalar lambda, const scalar mu, const scalar damping,
    int *nb_owners, // nb_vertices
    int *owner_off, // nb_vertices
    int *owners, // nb_neighbors.size()
    int *ref_vid, // nb_neighbors.size()
    int *topology, // nb_element * elem_nb_vert
    Vector3 *p, // nb_vertices
    Vector3 *p_init, // nb_vertices
    Vector3 *v, Vector3 *f,
    Vector3 *dN, // elem_nb_verts * nb_quadrature
    Matrix3x3 *JX_inv, // nb_element * nb_quadrature
    scalar *V // nb_element * nb_quadrature
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
    // and the number of needed quadratures
    const int cid = blockIdx.x; // vertex position
    const int size_of_block = nb_owners[cid] * nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % nb_quadrature; // quadrature number
    const int e_off = owner_off[cid] + tid / nb_quadrature; // offset in buffer to find the right element
    const int eid = owners[e_off]; // element id
    const int r_vid = ref_vid[e_off]; // vertex id in this element

    const int *topo = topology + eid * elem_nb_verts; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    const int qe_off = eid * nb_quadrature + qid;
    const int qv_off = qid * elem_nb_verts;

    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[9];
    Matrix3x3 P;

    for (int i = 0; i < elem_nb_verts; ++i) {
        Jx += glm::outerProduct(p[topo[i]], dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * JX_inv[qe_off];

    eval_stress(material, lambda, mu, F, P);
    eval_hessian(material, lambda, mu, F, d2W_dF2);

    const Vector3 dF_dx = glm::transpose(JX_inv[qe_off]) * dN[qv_off + r_vid];
    // Compute force at vertex i
    Vector3 fi = -P * dF_dx * V[qe_off];

    // assemble hessian
    Matrix3x3 H;
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            Matrix3x3 H_kl;
            for(int l = 0; l < 3; ++l) {
                for(int k = 0; k < 3; ++k) {
                    H_kl[k][l] = d2W_dF2[k+l*3][i][j];
                }
            }
            H[i][j] = glm::dot(dF_dx, H_kl * dF_dx) * V[qe_off];
        }
    }
    fi -= damping * H * v[vid];

    // shared variable : f, H
    __shared__ Vector3 s_f_H[256]; // size = block_size * 12 * sizeof(float)
    s_f_H[tid] = fi;

    __syncthreads();
    const int t = size_of_block;
    int i,b;
    for(i=t/2, b=(t+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            s_f_H[tid] += s_f_H[tid+b];
            __syncthreads();
        }
        i = (b>i) ? b : i;
    }

    if (threadIdx.x == 0) {
        f[vid] = Vector3(s_f_H[0]);
    }
}

void GPU_Explicit_FEM::step(const GPU_ParticleSystem* ps, scalar dt)
{
    int grid_size = (_nb_threads+_block_size-1)/_block_size;

    kernel_solve<<<grid_size, _block_size>>>(
        _nb_threads, nb_quadrature, elem_nb_vert, _material, lambda, mu, _damping,
        cb_nb_neighbors->buffer, cb_neighbors_offset->buffer, cb_neighbors->buffer, cb_ref_vid->buffer,
        cb_topology->buffer, ps->buffer_position(), ps->buffer_init_position(), ps->buffer_velocity(), ps->buffer_forces(),
        cb_dN->buffer, cb_JX_inv->buffer, cb_V->buffer
    );
}