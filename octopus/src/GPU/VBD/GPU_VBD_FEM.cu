#include "GPU/CUMatrix.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include "GPU/VBD/GPU_VBD_FEM.h"

#include <random>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/GPU_ParticleSystem.h>

__global__ void kernel_solve(
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
    Matrix3x3 d2W_dF2[9];

    for (int i = 0; i < fem.elem_nb_vert; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], fem.dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * fem.JX_inv[qe_off];
    Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    eval_hessian(mt.material, mt.lambda, mt.mu, F, d2W_dF2);


    const Vector3 dF_dx = glm::transpose(fem.JX_inv[qe_off]) * fem.dN[qv_off + r_vid];
    // Compute force at vertex i
    Vector3 fi = -P * dF_dx * fem.V[qe_off];

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
            H[i][j] = glm::dot(dF_dx, H_kl * dF_dx) * fem.V[qe_off];
        }
    }

    // shared variable : f, H
    // we can do a much better reduction (without atomic add with a shared buffer)

    __shared__ scalar s_f_H[2592]; // size = block_size * 12 * sizeof(float)
    for(int j = 0; j < 3; ++j) {
        s_f_H[tid * 12 + j] = fi[j];
        for(int i = 0; i < 3; ++i) {
            s_f_H[tid * 12 + (j+1)*3 + i] = H[i][j];
        }
    }

    __syncthreads();
    int t = size_of_block;
    int i,b;
    for(i=t/2, b=(t+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < 12; ++j) {
                s_f_H[tid*12+j] += s_f_H[(tid+b)*12+j];
            }
            __syncthreads();
        }
        i = (b>i) ? b : i;
    }

    if (threadIdx.x == 0) {
        H[0][0] = s_f_H[3]; H[0][1] = s_f_H[6]; H[0][2] = s_f_H[9];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[7]; H[1][2] = s_f_H[10];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[8]; H[2][2] = s_f_H[11];
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];

        // damping (velocity)
        fi -= damping / dt * H * (ps.p[vid] - ps.last_p[vid]);
        H  += damping / dt * H;

        // intertia (accellearation)
        scalar mh2 = ps.m[vid] / (dt*dt);
        fi -= mh2 * (ps.p[vid] - y[vid]);
        H[0][0] += mh2; H[1][1] += mh2; H[2][2] += mh2;

        //scalar detH = glm::determinant(s_H);
        //Vector3 dx = detH > 1e-6f ? glm::inverse(s_H) * s_f : Vector3(0.f);

        scalar detH = abs(glm::determinant(H));
        Vector3 dx = detH > 1e-6f ? glm::inverse(H) * fi : Vector3(0.f);
        ps.p[vid] += dx;
    }
}

void GPU_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937());

    for(const int c : kernels) {
        kernel_solve<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
            d_thread->nb_threads[c], _damping, dt, d_thread->offsets[c],
             y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }
}