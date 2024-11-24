#include "GPU/CUMatrix.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include "GPU/GPU_VBD_FEM.h"

#include <GPU/GPU_ParticleSystem.h>

__global__ void kernel_solve(
    // nb_thread, nb quadrature per elements, nb vertices in element
    const int n, const int nb_quadrature, const int elem_nb_verts,
    const scalar lambda, const scalar mu, const scalar damping, const scalar dt,
    int offset,
    int *nb_owners, // nb_vertices
    int *owner_off, // nb_vertices
    int *owners, // nb_neighbors.size()
    int *ref_vid, // nb_neighbors.size()
    int *topology, // nb_element * elem_nb_vert
    Vector3 *y, // nb_vertices
    Vector3 *p, // nb_vertices
    Vector3 *prev_p, // nb_vertices
    Vector3 *f,
    scalar *mass, // nb_vertices
    Vector3 *dN, // elem_nb_verts * nb_quadrature
    Matrix3x3 *JX_inv, // nb_element * nb_quadrature
    scalar *V // nb_element * nb_quadrature
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
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

    // shared variable : f, H
    // we can do a much better reduction (without atomic add with a shared buffer)

    __shared__ __builtin_align__(16) scalar s_f_H[1024]; // size = block_size * 12 * sizeof(float)
    for(int i = 0; i < 3; ++i) {
        s_f_H[tid * 12 + i] = fi[i];
        //s_f_H[tid * 12 + i] = 1;
        for(int j = 0; j < 3; ++j) {
            s_f_H[tid * 12 + (i+1)*3 + j] = H[i][j];
            //s_f_H[tid * 12 + (i+1)*3 + j] = 1;
        }
    }
    //printf("%d < %d\n", tid, size_of_block);

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

        // damping
        fi += -(damping / dt) * H * (p[vid] - prev_p[vid]);
        H += damping / dt * H;

        // intertia
        scalar mh2 = mass[vid] / (dt*dt);
        fi += -mh2 * (p[vid] - y[vid]);
        H[0][0] += mh2;
        H[1][1] += mh2;
        H[2][2] += mh2;

        //scalar detH = glm::determinant(s_H);
        //Vector3 dx = detH > 1e-6f ? glm::inverse(s_H) * s_f : Vector3(0.f);

        scalar detH = glm::determinant(H);
        Vector3 dx = detH > 1e-6f ? glm::inverse(H) * fi : Vector3(0.f);
        p[vid] += dx;
    }
}

void GPU_VBD_FEM::step(const GPU_ParticleSystem* ps, const scalar dt) {
    for(int c = 0; c < nb_color; ++c) {
        int grid_size = (c_nb_threads[c]+c_block_size[c]-1)/c_block_size[c];

        kernel_solve<<<grid_size, c_block_size[c]>>>(
        c_nb_threads[c], nb_quadrature, elem_nb_vert, lambda, mu, _damping, dt, c_offsets[c],
        cb_nb_neighbors->buffer, cb_neighbors_offset->buffer, cb_neighbors->buffer, cb_ref_vid->buffer,
        cb_topology->buffer,
        y->buffer, ps->buffer_position(),ps->buffer_prev_position(), ps->buffer_forces(), ps->buffer_mass(),
        cb_dN->buffer, cb_JX_inv->buffer, cb_V->buffer
        );
    }
}