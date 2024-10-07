#include "GPU/GPU_VBD.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/TimeManager.h>

__device__ Matrix3x3 vec_hat(const Vector3 &v) {
    return {
        0.f, -v.z, v.y,
        v.z, 0.f, -v.x,
        -v.y, v.x, 0.f
    };
}
__device__ void print_vec3(const Vector3 &v) {
    printf("(x:%f y:%f z:%f)", v.x, v.y, v.z);
}

__device__ void print_mat3(const Matrix3x3 &m) {
    printf("|%f %f %f|\n|%f %f %f|\n|%f %f %f|\n", m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2],
           m[2][2]);
}

__global__ void kernel_plane_fix(const int nb, scalar t, const Vector3 o, Vector3 n, Vector3 *p_init, Vector3 *y, Vector3 *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;

    Vector3 d = p_init[i] - o;
    if(glm::dot(d, n) > 0) {
        p[i] = p_init[i] + n * abs(cos(t+3.14f*0.5f)) * 0.f;
        y[i] = p[i];
    }
}

__global__ void kernel_integration(
        const int n, const scalar dt, const Vector3 g,
        Vector3 *p, Vector3 *prev_p, Vector3* y, Vector3* prev_it_p, Vector3 *v, Vector3 *f, scalar *w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    prev_p[i] = p[i]; // x^t-1 = x^t
    prev_it_p[i] = p[i];
    const Vector3 a_ext = g + f[i] * w[i];
    y[i] = p[i] + (v[i] + a_ext * dt) * dt;
    p[i] = y[i];

    f[i] *= 0;
}

__global__ void kernel_velocity_update(int n, scalar dt, Vector3* prev_p, Vector3* p, Vector3* v, scalar* _inv_mass) {
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= n) return;
    v[vid] = (p[vid] - prev_p[vid]) / dt;
    //if(vid == 10) printf("v(%f %f %f)\n", v[vid].x, v[vid].y, v[vid].z);
}

__global__ void kernel_chebychev_acceleration(int n, int it, scalar omega, Vector3* prev_it_p, Vector3* prev_it2_p, Vector3* p) {
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= n) return;
    if(it >= 2) {
        p[vid] = prev_it2_p[vid] + omega * (p[vid] - prev_it2_p[vid]);
    }
    prev_it2_p[vid] = prev_it_p[vid];
    prev_it_p[vid] = p[vid];
}

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



void GPU_VBD_FEM::step(const GPU_VBD* vbd, const scalar dt, const scalar damping) {
    for(int c = 0; c < nb_color; ++c) {
        int grid_size = (c_nb_threads[c]+c_block_size[c]-1)/c_block_size[c];

        kernel_solve<<<grid_size, c_block_size[c]>>>(
        c_nb_threads[c], nb_quadrature, elem_nb_vert, lambda, mu, damping, dt, c_offsets[c],
        cb_nb_neighbors->buffer, cb_neighbors_offset->buffer, cb_neighbors->buffer, cb_ref_vid->buffer,
        cb_topology->buffer,
        vbd->y->buffer, vbd->cb_position->buffer,vbd->cb_prev_position->buffer, vbd->cb_forces->buffer, vbd->cb_mass->buffer,
        cb_dN->buffer, cb_JX_inv->buffer, cb_V->buffer
        );
    }
}

void GPU_VBD::step(const scalar dt) const {
    Time::Tic();
    const scalar r = 0.8;
    const scalar sub_dt = dt / static_cast<scalar>(sub_iteration);
    Vector3 v = Unit3D::right();
    for(int i = 0; i < sub_iteration; ++i) {
        scalar omega = 1;
        // integration / first guess
        kernel_integration<<<(n + 255)/256, 256>>>(n,sub_dt,Dynamic::gravity(),
            cb_position->buffer,cb_prev_position->buffer,y->buffer, prev_it_p->buffer,
            cb_velocity->buffer,cb_forces->buffer, cb_inv_mass->buffer);

        for(int j = 0; j < iteration; ++j) {
            // solve
            dynamic->step(this, sub_dt, _damping);
            kernel_plane_fix<<<(n + 255)/256, 256>>>(n, Time::Fixed_Timer(), v*0.01f, -v, cb_init_position->buffer, y->buffer, cb_position->buffer);
            //kernel_plane_fix<<<(n + 255)/256, 256>>>(n, Time::Fixed_Timer(), v*1.99f, v, cb_init_position->buffer, y->buffer, cb_position->buffer);
            // Acceleration (Chebychev)
            if(j == 1) omega = 2.f / (2.f - r * r);
            else if(j > 1) omega = 4.f / (4.f - r * r * omega);
            //kernel_chebychev_acceleration<<<(n + 255)/256, 256>>>(n, j, omega, prev_it_p->buffer, prev_it2_p->buffer, cb_position->buffer);
        }
        // velocity update
        kernel_velocity_update<<<(n + 255)/256, 256>>>(n,sub_dt,
            cb_prev_position->buffer, cb_position->buffer, cb_velocity->buffer, cb_inv_mass->buffer);
    }
    cudaDeviceSynchronize();
    scalar time = Time::Tac() *1000.f;
    DebugUI::Begin("VBD");
    DebugUI::Plot("Time vbd", time);
    DebugUI::Value(" ", time);
    DebugUI::Range("", time);
    DebugUI::End();
}

GPU_VBD::~GPU_VBD() {
    delete integrator;
    delete dynamic;
}