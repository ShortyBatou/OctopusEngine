#include "GPU/VBD/GPU_LF_VBD_FEM.h"

#include <numeric>
#include <random>
#include <GPU/CUMatrix.h>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/Explicit/GPU_Explicit.h>
#include <Manager/TimeManager.h>

__device__ Matrix3x3 snh_lf_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu) {
    return mu * (F - mat3x3_com(F));
}

__device__ Matrix3x3 snh_lf_volume(const Matrix3x3 &F, const scalar lambda, const scalar mu) {
    return mat3x3_com(F);
}

__device__ void snh_lf_hessian(const Matrix3x3 &F, const scalar lambda, const scalar mu, Matrix3x3 d2W_dF2[6]) {
    // -mu * I3
    d2W_dF2[1] = vec_hat(F[2]) * -mu;
    d2W_dF2[2] = -vec_hat(F[1]) * -mu;
    d2W_dF2[4] = vec_hat(F[0]) * -mu;

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
    const Matrix3x3 P = snh_lf_stress(F, mt.lambda, mt.mu);
    Vector3 fi = -P * dF_dx * fem.V[qe_off];

    // Compute hessian
    snh_lf_hessian(F, mt.lambda, mt.mu, d2W_dF2);
    Matrix3x3 K = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2W_dF2);

    // damping (velocity)
    fi -= damping / dt * K * (ps.p[vid] - ps.last_p[vid]) * (1.f / scalar(owners.nb[cid]));
    K  += damping / dt * K * (1.f / scalar(owners.nb[cid]));

    // intertia (accellearation)
    scalar mh2 = ps.m[vid] / (dt*dt) * (1.f / scalar(owners.nb[cid]));
    fi -= mh2 * (ps.p[vid] - y[vid]);
    K[0][0] += mh2; K[1][1] += mh2; K[2][2] += mh2;

    Vector3 gradC = snh_lf_stress(F, mt.lambda, mt.mu) * dF_dx * fem.V[qe_off];

    // shared variable : f, H
    __shared__ scalar s_f_H[2592]; // size = block_size * 12 * sizeof(float)
    int k = 0;
    for(int j = 0; j < 3; ++j) {
        s_f_H[tid * 12 + j] = fi[j];
        s_f_H[tid * 12 + 3 + j] = gradC[j];
        for(int i = j; i < 3; ++i) {
            s_f_H[tid * 12 + 6 + k] = K[i][j];
            ++k;
        }
    }

    __syncthreads();
    int t = size_of_block;
    int i,j;
    for(i=t/2, j=(t+1)/2; i > 0; j=(j+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < 12; ++j) {
                s_f_H[tid*12+j] += s_f_H[(tid+j)*12+j];
            }
            __syncthreads();
        }
        i = (j>i) ? j : i;
    }

    if (threadIdx.x == 0) {
        Matrix4x4 A;
        Vector4 b;
        b.x = s_f_H[0]; b.y = s_f_H[1]; b.z = s_f_H[2];
        b.w = l[vid];

        A[0][0] = s_f_H[6];
        A[1][0] = s_f_H[7]; A[1][1] = s_f_H[8];
        A[2][0] = s_f_H[9]; A[2][1] = s_f_H[10];  A[2][2] = s_f_H[11];

        A[3][0] = s_f_H[3]; A[3][1] = s_f_H[4]; A[3][2] = s_f_H[5];
        A[3][3] = -1.f / mt.lambda;

        // symmetry
        A[0][1] = A[1][0]; A[1][2] = A[2][1]; A[0][2] = A[2][0];
        A[0][3] = A[3][0]; A[1][3] = A[3][1]; A[2][3] = A[3][2];

        //scalar detH = glm::determinant(s_H);
        scalar detH = abs(glm::determinant(A));
        Vector4 dx = detH > 1e-6f ? glm::inverse(A) * b : Vector4(0.f);
        ps.p[vid] += Vector3(dx.x, dx.y, dx.z);
        l[vid] += dx.w;
    }
}


void GPU_LF_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937());

    for(const int c : kernels) {
        kernel_lf_vbd_solve<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
             d_thread->nb_threads[c], _damping, dt, d_thread->offsets[c],
             y->buffer, l->buffer, *d_material,
             ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }
}