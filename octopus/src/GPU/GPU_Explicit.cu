#include "GPU/GPU_Explicit.h"

#include <GPU/CUMatrix.h>


__global__ void kernel_solve(
    // nb_thread, nb quadrature per elements, nb vertices in element
    const int n, const int nb_quadrature, const int elem_nb_verts,
    const scalar lambda, const scalar mu, scalar damping, scalar dt,
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
    //if(threadIdx.x == 0) printf("[%d][%d/%d] cid=%d, nb=%d, offset=%d, vid=%d, eid=%d, qid=%d, rid=%d, qe_off=%d, qv_off=%d \n",gid,tid+1,size_of_block,cid, nb_owners[cid],offset,vid, eid, qid, r_vid, qe_off, qv_off);
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[9];

    for (int i = 0; i < elem_nb_verts; ++i) {
        Jx += glm::outerProduct(p[topo[i]], dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * JX_inv[qe_off];


    /*
    //Hooke
    //force
    const Matrix3x3 e = 0.5f * (glm::transpose(F) + F ) - Matrix3x3(1.f);
    const Matrix3x3 P = lambda * (e[0][0]+e[1][1]+e[2][2]) * Matrix3x3(1.f) + mu * e;

    //Hessian
    for(int i = 0; i < 9; ++i) { d2W_dF2[i] = Matrix3x3(0); }
    for(int i = 0; i < 3; ++i) {
        d2W_dF2[i * 4] = Matrix3x3(lambda + mu);
    }
    */

    // Neohooke
    // Force
    const scalar detF = glm::determinant(F);
    const scalar alpha = 1.f + mu / lambda;
    Matrix3x3 comF(0);
    comF[0] = glm::cross(F[1], F[2]);
    comF[1] = glm::cross(F[2], F[0]);
    comF[2] = glm::cross(F[0], F[1]);
    const Matrix3x3 P = mu * F + lambda * (detF - alpha) * comF;
    // H = sum mi / h^2 I + sum d^2W / dxi^2
    scalar s = lambda * (detF - alpha);
    // lambda * (I3 - alpha) * H3
    d2W_dF2[0] = Matrix3x3(0);
    d2W_dF2[1] = vec_hat(F[2]) * s;
    d2W_dF2[2] = -vec_hat(F[1]) * s;
    d2W_dF2[3] = -d2W_dF2[1];
    d2W_dF2[4] = Matrix3x3(0);
    d2W_dF2[5] = vec_hat(F[0]) * s;
    d2W_dF2[6] = -d2W_dF2[2];
    d2W_dF2[7] = -d2W_dF2[5];
    d2W_dF2[8] = Matrix3x3(0);

    // mu/2 * H2 = mu * I_9x9x
    for (int i = 0; i < 3; ++i) {
        d2W_dF2[0][i][i] += mu;
        d2W_dF2[4][i][i] += mu;
        d2W_dF2[8][i][i] += mu;
    }

    // lambda vec(com F) * vec(com F)^T
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            d2W_dF2[i*3 + j] += glm::outerProduct(comF[i], comF[j]) * lambda;

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
    int t = size_of_block;
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
        _nb_threads, nb_quadrature, elem_nb_vert, lambda, mu, _damping, dt,
        cb_nb_neighbors->buffer, cb_neighbors_offset->buffer, cb_neighbors->buffer, cb_ref_vid->buffer,
        cb_topology->buffer, ps->buffer_position(), ps->buffer_init_position(), ps->buffer_velocity(), ps->buffer_forces(),
        cb_dN->buffer, cb_JX_inv->buffer, cb_V->buffer
    );
}