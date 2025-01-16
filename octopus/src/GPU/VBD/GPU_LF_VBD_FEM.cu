#include "GPU/VBD/GPU_LF_VBD_FEM.h"

#include <numeric>
#include <random>
#include <GPU/CUMatrix.h>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/Explicit/GPU_Explicit.h>
#include <Manager/TimeManager.h>

__device__ Matrix3x3 snh_lf_stress(const Matrix3x3 &F, const scalar l, const scalar mu) {
    return mu * (F) + 0.25f * l * mat3x3_com(F);
}

__device__ void snh_lf_volume(const Matrix3x3 &F, Matrix3x3& P, scalar& C) {
    C = glm::determinant(F) - 1;
    P = mat3x3_com(F);
}

__device__ void snh_lf_hessian(const Matrix3x3 &F, const scalar l, const scalar mu, Matrix3x3 d2W_dF2[6]) {
    // (0.25l - mu) * (d^2_I3 / dx^2)
    d2W_dF2[1] = vec_hat(F[2])  * (0.25f * l );
    d2W_dF2[2] = -vec_hat(F[1]) * (0.25f * l );
    d2W_dF2[4] = vec_hat(F[0])  * (0.25f * l );

    // mu/2 * H2 = mu * I_9x9x
    d2W_dF2[0] = Matrix3x3(mu);
    d2W_dF2[3] = Matrix3x3(mu);
    d2W_dF2[5] = Matrix3x3(mu);
}


__global__ void kernel_lf_vbd_solve(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    scalar* l,
    const scalar* Vi,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
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
    //if(threadIdx.x == 0) printf("[%d][%d/%d] cid=%d, nb=%d, offset=%d, vid=%d, eid=%d, qid=%d, rid=%d, qe_off=%d, qv_off=%d \n",gid,tid+1,size_of_block,cid, nb_owners[cid],offset,vid, eid, qid, r_vid, qe_off, qv_off);
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[6];

    for (int i = 0; i < fem.elem_nb_vert; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], fem.dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * fem.JX_inv[qe_off];
    const Vector3 dF_dx = glm::transpose(fem.JX_inv[qe_off]) * fem.dN[qv_off + r_vid];

    // Compute force at vertex i
    //Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    Matrix3x3 P = snh_lf_stress(F, l[vid], mt.mu);
    Vector3 fi = -P * dF_dx * fem.V[qe_off];

    // Compute hessian
    snh_lf_hessian(F, l[vid], mt.mu, d2W_dF2);
    Matrix3x3 K = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2W_dF2);

    // damping (velocity)
    fi -= damping / (dt*size_of_block) * K * (ps.p[vid] - ps.last_p[vid]);
    K  += damping / (dt*size_of_block) * K;

    // intertia (accellearation)
    scalar mh2 = ps.m[vid] / (dt*dt*size_of_block);
    fi -= mh2 * (ps.p[vid] - y[vid]);
    K[0][0] += mh2; K[1][1] += mh2; K[2][2] += mh2;

    scalar C = 0;
    snh_lf_volume(F, P, C);
    Vector4 gradC(  P * dF_dx, C);
    gradC *= 0.25f * fem.V[qe_off];
    // shared variable : f, H
    __shared__ scalar s_f_H[2592]; // size = block_size * 12 * sizeof(float)
    s_f_H[tid * 13 + 6] = gradC.w;
    for(int k = 0, j = 0; j < 3; ++j) {
        s_f_H[tid * 13 + j] = fi[j];
        s_f_H[tid * 13 + 3 + j] = gradC[j];
        for(int i = j; i < 3; ++i) {
            s_f_H[tid * 13 + 7 + k] = K[i][j];
            ++k;
        }
    }

    __syncthreads();
    int t = size_of_block;
    for(int i=t/2, k=(t+1)/2; i > 0; k=(k+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < 13; ++j) {
                s_f_H[tid*13+j] += s_f_H[(tid+k)*13+j];
            }
            __syncthreads();
        }
        i = (k>i) ? k : i;
    }

    if (threadIdx.x == 0) {
        Matrix4x4 A(1.);
        Vector4 b(s_f_H[0], s_f_H[1], s_f_H[2], s_f_H[6]);
        A[0][0] = s_f_H[7];
        A[1][0] = s_f_H[8]; A[1][1] = s_f_H[10];
        A[2][0] = s_f_H[9]; A[2][1] = s_f_H[11]; A[2][2] = s_f_H[12];
        A[3][0] = s_f_H[3]; A[3][1] = s_f_H[4]; A[3][2] = s_f_H[5];
        A[3][3] = -Vi[vid] / mt.lambda;

        // symmetry
        A[0][1] = A[1][0]; A[1][2] = A[2][1]; A[0][2] = A[2][0];
        A[0][3] = A[3][0]; A[1][3] = A[3][1]; A[2][3] = A[3][2];

        //scalar detH = glm::determinant(s_H);
        scalar detH = abs(glm::determinant(A));
        Vector4 dx = detH > 1e-6f ? glm::inverse(A) * b : Vector4(0.f);
        ps.p[vid] += Vector3(dx[0], dx[1], dx[2]);
        l[vid] += dx[3];

        /*Matrix3x3 K_inv;
        K_inv[0][0] = s_f_H[7];
        K_inv[1][0] = s_f_H[8]; K_inv[1][1] = s_f_H[10];
        K_inv[2][0] = s_f_H[9]; K_inv[2][1] = s_f_H[11]; K_inv[2][2] = s_f_H[12];
        K_inv[0][1] = K_inv[1][0]; K_inv[1][2] = K_inv[2][1]; K_inv[0][2] = K_inv[2][0];

        K_inv = glm::inverse(K_inv);

        Vector3 G(s_f_H[3], s_f_H[4], s_f_H[5]);
        Vector3 f(s_f_H[0], s_f_H[1], s_f_H[2]);

        scalar dt_l = (-s_f_H[6] + glm::dot(G, K_inv*f)) / (Vi[vid] / mt.lambda + glm::dot(G, K_inv * G));
        Vector3 dt_p = K_inv * G * dt_l;
        ps.p[vid] += dt_p;
        l[vid] += dt_l;*/
    }
}


__global__ void kernel_reset(const int n, scalar* l)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= n) return;
    l[gid] = 0.f;
}

void GPU_LF_VBD_FEM::start(GPU_ParticleSystem* ps, scalar dt)
{
    const int nb_thread = ps->nb_particles();
    const int block_size = 32;
    const int grid_size = (nb_thread + block_size-1) / block_size;
    kernel_reset<<<grid_size, block_size>>>(nb_thread, l->buffer);
}

void GPU_LF_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937());

    for(const int c : kernels) {
        kernel_lf_vbd_solve<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
             d_thread->nb_threads[c], _damping, dt, d_thread->offsets[c],
             y->buffer, l->buffer, Vi->buffer, *d_material,
             ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }
}

