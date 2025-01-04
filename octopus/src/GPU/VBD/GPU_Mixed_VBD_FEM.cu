#include "GPU/VBD/GPU_Mixed_VBD_FEM.h"

#include <GPU/GPU_FEM_Material.h>
#include <GPU/Explicit/GPU_Explicit.h>


__global__ void kernel_explicit_fem_eval_force_2(
    // nb_thread, nb quadrature per elements, nb vertices in element
    const int n, const scalar damping,
    const Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners,
    scalar* w_max
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
    // and the number of needed quadratures
    const int cid = blockIdx.x; // vertex position
    const int size_of_block = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;

    Matrix3x3 Jx(0.f);

    for (int i = 0; i < fem.elem_nb_vert; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], fem.dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * fem.JX_inv[qe_off];

    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);

    const Vector3 dF_dx = glm::transpose(fem.JX_inv[qe_off]) * fem.dN[qv_off + r_vid];
    // Compute force at vertex i
    Vector3 fi = -P * dF_dx * fem.V[qe_off];

    // assemble hessian
    Matrix3x3 d2W_dF2[6];
    eval_hessian(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    Matrix3x3 H = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2W_dF2);

    //Matrix3x3 H2 = glm::outerProduct(fi, fi);
    fi -= damping * H * ps.v[vid];/**/

    scalar l_w_max = 0;
    for(int i = 0; i < 3; ++i)
    {
        scalar w = 0;
        for(int j = 0; j <3; ++j)
        {
            w+= fabsf(H[i][j]);
        }
        l_w_max = fmaxf(l_w_max, w);
    }

    // shared variable : f, H
    __shared__ Vector3 s_f_H[256]; // size = block_size * 3 * sizeof(float)
    __shared__ scalar s_w_max[64]; // size = block_size * sizeof(float)
    s_f_H[tid] = fi;
    s_w_max[tid] = l_w_max;

    __syncthreads();
    const int t = size_of_block;
    int i,b;
    for(i=t/2, b=(t+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            s_f_H[tid] += s_f_H[tid+b];
            s_w_max[tid] += s_w_max[tid+b] ;
            __syncthreads();
        }
        i = (b>i) ? b : i;
    }

    if (threadIdx.x == 0) {
        ps.f[vid] = s_f_H[0];
        w_max[vid] = s_w_max[0] * ps.w[vid];
    }
}

__global__ void kernel_explicit_fem_eval_force_3(
    // nb_thread, nb quadrature per elements, nb vertices in element
    const int n,
    const Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
    // and the number of needed quadratures
    const int cid = blockIdx.x; // vertex position
    const int size_of_block = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;

    Matrix3x3 Jx(0.f);
    for (int i = 0; i < fem.elem_nb_vert; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], fem.dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * fem.JX_inv[qe_off];
    const Vector3 dF_dx = glm::transpose(fem.JX_inv[qe_off]) * fem.dN[qv_off + r_vid];
    // Compute force at vertex i
    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    const Vector3 fi = -P * dF_dx * fem.V[qe_off];

    // shared variable : f, H
    __shared__ Vector3 s_f_H[256]; // size = block_size * 3 * sizeof(float)
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
        ps.f[vid] = s_f_H[0];
    }
}

void GPU_Mixed_VBD_FEM::explicit_step(GPU_ParticleSystem* ps, Cuda_Buffer<scalar>* w_max, scalar dt) const
{
     /*kernel_explicit_fem_eval_force_2<<<d_exp_thread->grid_size[0], d_exp_thread->block_size[0]>>>(
        d_exp_thread->nb_threads[0], _damping, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters(), w_max->buffer
    );/**/


    kernel_explicit_fem_eval_force_3<<<d_exp_thread->grid_size[0], d_exp_thread->block_size[0]>>>(
        d_exp_thread->nb_threads[0], *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
    );/**/
}
