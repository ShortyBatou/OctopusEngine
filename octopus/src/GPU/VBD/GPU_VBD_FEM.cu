#include "GPU/CUMatrix.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include "GPU/VBD/GPU_VBD_FEM.h"

#include <random>
#include <set>
#include <Tools/Graph.h>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/GPU_ParticleSystem.h>
#include <Manager/Input.h>
#include <Manager/TimeManager.h>

__device__ void compute_f_H(
    const int n, const int r_vid,
    const Material_Data& mt, const GPU_ParticleSystem_Parameters ps, const int* topo,
    const Matrix3x3 &JX_inv, const scalar V, const Vector3* dN,
    Vector3& fi, Matrix3x3& H) {
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[9];

    for (int i = 0; i < n; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], dN[i]);
    }

    const Matrix3x3 F = Jx * JX_inv;
    const Vector3 dF_dx = glm::transpose(JX_inv) * dN[r_vid];

    // Compute force at vertex i
    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    fi -= P * dF_dx * V;

    // Compute hessian
    eval_hessian(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    H += assemble_sub_hessian(dF_dx, V, d2W_dF2);
}


__device__ void compute_f_H_sym(
    const int n, const int r_vid,
    const Material_Data& mt, const Vector3* p, const int* topo,
    const Matrix3x3 &JX_inv, const scalar V, const Vector3* dN,
    Vector3& fi, Matrix3x3& H) {
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[6];

    for (int i = 0; i < n; ++i) {
        Jx += glm::outerProduct(p[topo[i]], dN[i]);
    }

    const Matrix3x3 F = Jx * JX_inv;
    const Vector3 dF_dx = glm::transpose(JX_inv) * dN[r_vid];

    // Compute force at vertex i
    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    fi -= P * dF_dx * V;

    // Compute hessian
    eval_hessian_sym(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    H += assemble_sub_hessian_sym(dF_dx, V, d2W_dF2);
}

__device__ void compute_f(
    const int n, const int r_vid,
    const Material_Data& mt, const Vector3* p, const int* topo,
    const Matrix3x3 &JX_inv, const scalar V, const Vector3* dN,
    Vector3& fi) {
    Matrix3x3 Jx(0.f);

    for (int i = 0; i < n; ++i) {
        Jx += glm::outerProduct(p[topo[i]], dN[i]);
    }

    const Matrix3x3 F = Jx * JX_inv;
    const Vector3 dF_dx = glm::transpose(JX_inv) * dN[r_vid];

    // Compute force at vertex i
    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    fi -= P * dF_dx * V;
}

__device__ void store_f_H_in_shared(const int tid, const Vector3& fi, const Matrix3x3& H, scalar* s_data) {
    int k = 0;
    for(int j = 0; j < 3; ++j) {
        s_data[tid * 12 + j] = fi[j];
        for(int i = 0; i < 3; ++i) {
            s_data[tid * 12 + 3 + k] = H[i][j];
            ++k;
        }
    }
}

__device__ void store_f_H_in_shared_sym(const int tid, const Vector3& fi, const Matrix3x3& H, scalar* s_data) {
    int k = 0;
    for(int j = 0; j < 3; ++j) {
        s_data[tid * 9 + j] = fi[j];
        for(int i = j; i < 3; ++i) {
            s_data[tid * 9 + 3 + k] = H[i][j];
            ++k;
        }
    }
}

__device__ Vector3 compute_correction(const int vid, const scalar damping, const scalar dt,
        GPU_ParticleSystem_Parameters ps, const Vector3* y,
        Vector3& fi, Matrix3x3& H) {
    // damping (velocity)
    fi -= damping / dt * H * (ps.p[vid] - ps.last_p[vid]);
    H  += damping / dt * H;

    // inertia (acceleration)
    const scalar mh2 = ps.m[vid] / (dt*dt);
    fi -= mh2 * (ps.p[vid] - y[vid]);
    H[0][0] += mh2; H[1][1] += mh2; H[2][2] += mh2;

    //scalar detH = glm::determinant(s_H);
    const scalar detH = abs(glm::determinant(H));
    return detH > 1e-6f ? glm::inverse(H) * fi : Vector3(0.f);
}

__global__ void kernel_vbd_solve_v1(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    if (blockIdx.x * blockDim.x + threadIdx.x >= n) return;

    // the group size depends on the number of element that contains these vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
    const int block_size = owners.nb[cid];
    const int tid = threadIdx.x;
    if (tid >= block_size) return;

    const int e_off = owners.offset[cid] + tid; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];

    if(ps.mask[vid] == 0) return;

    Vector3 fi(0.f);
    Matrix3x3 H(0.f);
    for(int q = 0; q < fem.nb_quadrature; ++q) {
        const int qe_off = eid * fem.nb_quadrature + q;
        const int qv_off = q * fem.elem_nb_vert;
        compute_f_H(fem.elem_nb_vert, r_vid,
                    mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                    fi, H);
    }

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 12 * sizeof(float)
    store_f_H_in_shared(tid, fi, H, s_f_H); // store f and h in shared memory
    vec_reduction(tid, block_size, 0, 12, s_f_H); // reduction of fi Hi

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3]; H[0][1] = s_f_H[6]; H[0][2] = s_f_H[9 ];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[7]; H[1][2] = s_f_H[10];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[8]; H[2][2] = s_f_H[11];

        ps.p[vid] += compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}

__global__ void kernel_vbd_solve_v2(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
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
    const int block_size = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= block_size) return;

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

    compute_f_H(fem.elem_nb_vert, r_vid,
                    mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                    fi, H);

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 12 * sizeof(float)
    store_f_H_in_shared(tid, fi, H, s_f_H); // store f and h in shared memory
    vec_reduction(tid, block_size, 0, 12, s_f_H); // reduction of fi Hi

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3]; H[0][1] = s_f_H[6]; H[0][2] = s_f_H[9 ];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[7]; H[1][2] = s_f_H[10];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[8]; H[2][2] = s_f_H[11];

        ps.p[vid] += compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}

__global__ void kernel_vbd_solve_v3(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
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
        ps.p[vid] += compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}


__global__ void kernel_vbd_solve_v3_test(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
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
    const int cid = blockIdx.x; // vertex position in coloration
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

    compute_f(fem.elem_nb_vert, r_vid,
                mt, ps.p, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                fi);

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 9 * sizeof(float)
    for(int j = 0; j < 3; ++j) {
        s_f_H[tid * 3 + j] = fi[j];
    }
    vec_reduction(tid, size_of_block, 0, 3, s_f_H);

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        //fi -= ps.m[vid] / (dt*dt) * (ps.p[vid] - y[vid]);
        Matrix3x3 H(dt*dt / (dt*dt + ps.m[vid]));
        ps.p[vid] += H * fi;
    }
}

__global__ void kernel_vbd_solve_v4(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
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

__global__ void kernel_copy_in_position(const int n, const int offset, const Vector3* p_new,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem, GPU_Owners_Parameters owners) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const int cid = offset + gid;
    const int e_off = owners.offset[cid];
    const int r_vid = owners.ref_vid[e_off];
    const int *topo = fem.topology + owners.eids[e_off] * fem.elem_nb_vert;
    const int vid = topo[r_vid];
    if(ps.mask[vid] == 0) return;
    ps.p[vid] = p_new[vid];
}

__global__ void kernel_vbd_solve_v5(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners,
    GPU_BLock_Parameters blocks
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // sub block system
    const int bid = offset + blockIdx.x; // block id
    const int b_sub_size = blocks.sub_block_size[bid];
    const int b_nb_sub_block = blocks.nb_sub_block[bid];
    const int b_size = b_sub_size * b_nb_sub_block;
    const int b_sub_id = threadIdx.x / b_sub_size;
    if(threadIdx.x >= b_size) return;


    const int tid = threadIdx.x % b_sub_size;
    // vertex information
    const int cid = blocks.offset[bid] + b_sub_id;
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
    const int s_off = b_sub_id * b_sub_size;
    const int o = s_off * 9;
    store_f_H_in_shared_sym(tid, fi, H, s_f_H+o);
    vec_reduction(tid, b_sub_size, s_off, 9, s_f_H);

    if (tid == 0) {
        fi.x = s_f_H[o + 0]; fi.y = s_f_H[o + 1]; fi.z = s_f_H[o + 2];
        H[0][0] = s_f_H[o + 3];
        H[1][0] = s_f_H[o + 4]; H[1][1] = s_f_H[o + 6];
        H[2][0] = s_f_H[o + 5]; H[2][1] = s_f_H[o + 7];  H[2][2] = s_f_H[o + 8];
        // symmetry
        H[0][1] = H[1][0]; H[1][2] = H[2][1]; H[0][2] = H[2][0];

        ps.p[vid] += compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}


__global__ void kernel_vbd_compute_residual(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    Vector3* r,
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
    compute_f_H(fem.elem_nb_vert, r_vid,
                mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                fi, H);

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 12 * sizeof(float)
    store_f_H_in_shared_sym(tid, fi, H, s_f_H);
    vec_reduction(tid, size_of_block, 0, 9, s_f_H);

    if (threadIdx.x == 0) {
        fi = Vector3(0);
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[6];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[7];  H[2][2] = s_f_H[8];
        // symmetry
        H[0][1] = H[1][0]; H[1][2] = H[2][1]; H[0][2] = H[2][0];

        // damping (velocity)
        fi -= damping / dt * H * (ps.p[vid] - ps.last_p[vid]);
        H  += damping / dt * H;

        // intertia (accellearation)
        const scalar mh2 = ps.m[vid] / (dt*dt);
        fi -= mh2 * (ps.p[vid] - y[vid]);
        H  += mh2 * Matrix3x3(1.);
        r[vid] = fi;
    }
}

std::vector<Vector3> GPU_VBD_FEM::get_forces(const GPU_ParticleSystem *ps, const scalar dt) const {
    for(int c = 0; c < d_thread->nb_kernel; ++c) {
        scalar s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
        kernel_vbd_compute_residual<<<d_thread->grid_size[c] * d_fem->nb_quadrature, d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
            d_thread->nb_threads[c], damping, dt, d_thread->offsets[c],
             y->buffer, r->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }
    std::vector<Vector3> residual(ps->nb_particles());
    r->get_data(residual);
    return residual;
}


GPU_VBD_FEM::GPU_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping,
                         const VBD_Version& v) :
    GPU_FEM(element, geometry, topology, young, poisson, material), version(v), y(nullptr)
{
    const int nb_vertices = static_cast<int>(geometry.size());
    this->damping = damping;
    d_owners = new GPU_Owners_Data();
    d_blocks = new GPU_Block_Data();
    r = new Cuda_Buffer(nb_vertices, Vector3(0.f));
    std::cout << "#ELEMENT = " << topology.size() / elem_nb_vertices(element) << std::endl;
    std::cout << "#VERTEX = " << nb_vertices << std::endl;
    std::cout << "#QUADRATURE " << d_fem->nb_quadrature << std::endl;
    std::vector<std::vector<int>> e_owners;
    std::vector<std::vector<int>> e_ref_id;
    build_owner_data(nb_vertices, topology, e_owners, e_ref_id);
    Coloration coloration = build_graph_color(element, topology); // get coloration
    create_buffers(element, topology, coloration, e_owners, e_ref_id);
}

void GPU_VBD_FEM::create_buffers(
    const Element element,
    const Mesh::Topology& topology,
    Coloration& coloration,
    std::vector<std::vector<int>>& e_owners,
    std::vector<std::vector<int>>& e_ref_id)
{

    std::vector<scalar> w;

    // sort by color
    std::vector<int> ref_id;
    std::vector<int> owners;
    std::vector<int> nb_owners;
    std::vector<int> owners_offset;

    // better coloration
    std::vector<scalar> weights;

    // blocking optimisation
    std::vector<int> nb_sub_block;      // nb sub_block in block
    std::vector<int> sub_block_size;    // size of one sub block
    std::vector<int> block_offset;      // first vertice id in block

    // need to remake topology and geometry to take in count ghost particles
    // get the mapping between the modified mesh and the true mesh
    if(version >= Better_Coloration) {
        Graph p_graph(element, topology); // redondant with coloration
        GraphReduction::MinimalColorationConflict(element, topology, p_graph, coloration, e_owners, e_ref_id);
        p_new = new Cuda_Buffer<Vector3>(std::vector<Vector3>(coloration.colors.size(), Unit3D::Zero()));
        std::map<int, int> _t_conflict = GraphColoration::Get_Conflict(p_graph, coloration);
        int nb_conflict = 0;
        for(auto [_, nb] : _t_conflict) {
            nb_conflict += nb;
        }
        std::cout << "NB CONFLICT : " << nb_conflict << std::endl;
        std::cout << "Coloration : nb = " << coloration.nb_color << std::endl;
    }

    int total_thread = 0;
    int total_merge_thread = 0;
    // reorder data to match coloration and get multi-threading data
    for(int c = 0; c < coloration.nb_color; ++c) {
        int n_max = 1; // max block size
        int nb_block = 0; // nb vert in color
        int offset = static_cast<int>(version == Block_Merge ? nb_sub_block.size() : nb_owners.size()); // nb vertices

        // get all ids in coloration and get the max block size
        std::vector<int> ids; // block id
        for(int i = 0; i < coloration.colors.size(); ++i) {
            if(c != coloration.colors[i]) continue;
            ids.push_back(i);
            n_max = std::max(n_max, static_cast<int>(e_owners[i].size()));
            nb_block++;
        }

        total_thread += nb_block * n_max;
        // the max block size depends on the largest block in color and needs to be a multiple of 32 (NVidia)
        int vmax = n_max;

        if(version == Block_Merge) {
            // on devrait avoir une taille par bloc !
            // s'il y a  32 thread qui font rien, le bloc est terminé extrèmement vite
            vmax = (n_max * d_fem->nb_quadrature / 32 + (n_max * d_fem->nb_quadrature % 32 == 0 ? 0 : 1)) * 32;
            nb_block = 0;
            // sort id by nb_owners
            std::sort(ids.begin(), ids.end(),
                 [&](const int& a, const int& b) {
                     return e_owners[a].size() < e_owners[b].size();
                 }
            );

            // create the block merge data
            std::cout << "C = " << c << "   MAX = " << vmax << std::endl;
            int off = nb_owners.size();
            for(int i = 0; i < ids.size(); ++i) {
                const int v = e_owners[ids[i]].size() * d_fem->nb_quadrature; // block size
                // nb block that have the same size
                int nb = 1;
                while(i+1 < ids.size() && e_owners[ids[i+1]].size() * d_fem->nb_quadrature == v) {
                    nb++; i++;
                }
                // how we merge the block depending on the max block size
                const int max_block = vmax / v; // max size sub_group
                const int nb_group = nb / max_block; // how many sub group
                const int rest = (nb - max_block * nb_group) * v;  // nb thread that can't be totally merge
                for(int j = 0; j < nb_group; ++j) {
                    nb_sub_block.push_back(max_block);
                    sub_block_size.push_back(v);
                    block_offset.push_back(off);
                    off += max_block;
                    nb_block++;
                }
                if(rest != 0) {
                    nb_sub_block.push_back(nb - max_block * nb_group);
                    sub_block_size.push_back(v);
                    block_offset.push_back(off);
                    off += nb - max_block * nb_group;
                    nb_block++;
                }
                std::cout << v << " x " << nb << " => " << nb_group << "x(" << max_block << "x" << v << ") + (" << rest / v << "x" << v << ")" << std::endl;
            }
            total_merge_thread += n_max * nb_block;
            std::cout << std::endl;
            vmax /= d_fem->nb_quadrature;
        }

        for(const int id : ids) {
            nb_owners.push_back(static_cast<int>(e_owners[id].size()));
            owners_offset.push_back(static_cast<int>(owners.size()));
            owners.insert(owners.end(), e_owners[id].begin(), e_owners[id].end());
            ref_id.insert(ref_id.end(), e_ref_id[id].begin(), e_ref_id[id].end());
        }

        // multi-threading data
        d_thread->grid_size.push_back(nb_block);
        d_thread->nb_threads.push_back(nb_block * vmax);
        d_thread->offsets.push_back(offset);
        d_thread->block_size.push_back(vmax);
    }
    std::cout << total_thread << " vs " << total_merge_thread << std::endl;
    d_thread->nb_kernel = coloration.nb_color;

    //vert data
    d_owners->cb_nb = new Cuda_Buffer(nb_owners);
    d_owners->cb_offset = new Cuda_Buffer(owners_offset);

    // element data
    d_owners->cb_eids = new Cuda_Buffer(owners);
    d_owners->cb_ref_vid = new Cuda_Buffer(ref_id);

    // block data
    d_blocks->cb_nb_sub_block = new Cuda_Buffer<int>(nb_sub_block);
    d_blocks->cb_offset = new Cuda_Buffer<int>(block_offset);
    d_blocks->cb_sub_block_size = new Cuda_Buffer<int>(sub_block_size);
}

void GPU_VBD_FEM::build_owner_data(
    const int nb_vertices, const Mesh::Topology &topology,
    std::vector<std::vector<int>>& e_neighbors, std::vector<std::vector<int>>& e_ref_id) const {
    e_neighbors.resize(nb_vertices);
    e_ref_id.resize(nb_vertices);
    // for each vertice get all its neighboors
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        int eid = i / d_fem->elem_nb_vert;
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            e_neighbors[topology[i + j]].push_back(eid);
            e_ref_id[topology[i+j]].push_back(j);
        }
    }
}


Coloration GPU_VBD_FEM::build_graph_color(const Element element, const Mesh::Topology &topology)
{
    Graph p_graph(element, topology);
    //Graph d_graph(element, topology, false);
    //Coloration coloration = GraphColoration::Primal_Dual_Element(element, topology, *p_graph, *d_graph);
    Coloration coloration = GraphColoration::DSAT(p_graph);
    return coloration;
}



void GPU_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937(seed));
    unsigned int s;


    for(const int c : kernels) {
        switch(version) {
            case Base :
                s = d_thread->block_size[c] * 12 * sizeof(scalar);
                kernel_vbd_solve_v1<<<d_thread->grid_size[c], d_thread->block_size[c], s>>>(
                    d_thread->nb_threads[c], damping, dt, d_thread->offsets[c],
                      y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters());
            break;
            case Threaded_Quadrature :
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 12 * sizeof(scalar);
                kernel_vbd_solve_v2<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                     y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
                );
            break;
            case Reduction_Symmetry :
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
                kernel_vbd_solve_v3<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                    y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
                );
            break;
            case Better_Coloration :
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
                kernel_vbd_solve_v4<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                    y->buffer, p_new->buffer, *d_material,
                    ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
                );
                kernel_copy_in_position<<<d_thread->grid_size[c] / 32 + 1, 32>>>(
                    d_thread->grid_size[c], d_thread->offsets[c], p_new->buffer,
                    ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
                );
                break;
            case Block_Merge :
                break;
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
                kernel_vbd_solve_v5<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                    y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters(), get_block_parameters()
                );
                break;
        }
    }
}