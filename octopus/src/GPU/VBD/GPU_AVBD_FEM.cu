#include <glm/detail/func_matrix_simd.inl>
#include "GPU/VBD/GPU_AVBD_FEM.h"

#include <GPU/CUMatrix.h>

__global__ void kernel_init_constraint(
    const int n, scalar max_k, scalar alpha, scalar gamma,
    scalar* lambda,
    scalar* k
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    lambda[gid] *= alpha * gamma;
    k[gid] = max(max_k, gamma * k[gid]);
}



__global__ void kernel_avbd_solve_distance(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const scalar lambda_min, const scalar lambda_max,
    const Vector3* y,
    const scalar* lambda,
    const scalar* k,
    Vector3* p_new,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains these vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
    const int size_of_block = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    if(ps.mask[vid] == 0) return;

    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;

    Vector3 fi(0.f);
    Matrix3x3 H(0.f);

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 9 * sizeof(float)
    store_f_H_in_shared_sym(tid, fi, H, s_f_H);
    vec_reduction(tid, size_of_block, 0, 9, s_f_H);

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[6];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[7];  H[2][2] = s_f_H[8];
        // symmetry
        H[0][1] = H[1][0]; H[1][2] = H[2][1]; H[0][2] = H[2][0];
        p_new[vid] = ps.p[vid] + compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}


__global__ void kernel_avbd_solve(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    const scalar* lambda,
    const scalar* k,
    Vector3* p_new,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains these vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
    const int size_of_block = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    if(ps.mask[vid] == 0) return;

    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;
    Vector3 fi(0.f);
    Matrix3x3 H(0.f);
    compute_f_H_sym(fem.elem_nb_vert, r_vid,
                mt, ps.p, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                fi, H);


    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 9 * sizeof(float)
    store_f_H_in_shared_sym(tid, fi, H, s_f_H);
    vec_reduction(tid, size_of_block, 0, 9, s_f_H);

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[6];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[7];  H[2][2] = s_f_H[8];
        // symmetry
        H[0][1] = H[1][0]; H[1][2] = H[2][1]; H[0][2] = H[2][0];
        p_new[vid] = ps.p[vid] + compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}