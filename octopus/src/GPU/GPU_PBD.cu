#include "GPU/GPU_PBD.h"

#include <Manager/Debug.h>

__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3 *pos, Vector3 *dN) {
    // Compute transform (reference => scene)
    Matrix3x3 Jx = Matrix3x3(0.);;
    for (int j = 0; j < nb_vert_elem; ++j) {
        Jx = glm::outerProduct(pos[j], dN[nb_vert_elem + j]);
    }

    return Jx;
}


__global__ void kernel_velocity_update(const int n, const float dt, Vector3 *p, Vector3 *prev_p, Vector3 *v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = (p[i] - prev_p[i]) / dt;
}


__global__ void kernel_step_solver(const int n, const float dt, const Vector3 g, Vector3 *p, Vector3 *prev_p,
                                   Vector3 *v, Vector3 *f, float *w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    prev_p[i] = p[i];
    v[i] += (g + f[i] * w[i]) * dt;
    p[i] += v[i] * dt;
    f[i] *= 0;
}

__global__ void kernel_constraint_plane(const int n, const Vector3 origin, const Vector3 normal, Vector3 *p,
                                        Vector3 *p_init) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vector3 d = p_init[i] - origin;
    scalar s = dot(d, normal);
    if (s > 0) {
        p[i] = p_init[i];
    }
}

__device__ scalar mat3x3_trace(const Matrix3x3 &m) {
    return m[0][0] + m[1][1] + m[2][2];
}


__device__ void print_vec(const Vector3 &v) {
    printf("x:%f y:%f z:%f", v.x, v.y, v.z);
}

__device__ void print_mat(const Matrix3x3 &m) {
    printf("%f %f %f %f %f %f %f %f %f", m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2],
           m[2][2]);
}

__device__ void hooke_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    scalar trace = mat3x3_trace(0.5f * (glm::transpose(F) * F - Matrix3x3(1.f)));
    C = trace * trace;
    P = (2.f * trace * F);
}

__device__ void hooke_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    Matrix3x3 E = 0.5f * (glm::transpose(F) * F - Matrix3x3(1.f));
    P = 2.f * F * E;
    C = (E[0][0] * E[0][0] + E[1][1] * E[1][1] + E[2][2] * E[2][2]) + (
            E[0][1] * E[1][0] + E[1][2] * E[2][1] + E[2][0] * E[0][2]) * 2.f;
}

__global__ void kernel_constraint_solve_v0(
    const int n, const int nb_quadrature, const int nb_vert_elem, const scalar dt, // some global data
    const int material, const scalar stiffness, // material
    const int offset, // coloration
    Vector3* cb_dN,
    Vector3 *cb_p, int *cb_topology, // mesh
    scalar *inv_mass,
    scalar *cb_V, Matrix3x3 *cb_JX_inv // element data (Volume * Weight, Inverse of initial jacobian)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    int eid = tid + offset / nb_vert_elem;
    int vid = tid * nb_vert_elem + offset; // first vertice id in topology

    Vector3 grad_C[32];
    for (int j = 0; j < nb_vert_elem; ++j)
        grad_C[j] = Vector3(0, 0, 0);

    scalar C = 0.f;
    for (int q = 0; q < nb_quadrature; ++q) {
        Matrix3x3 JXinv = cb_JX_inv[eid * nb_quadrature + q];
        scalar V = cb_V[eid * nb_quadrature + q];

        const int q_off = nb_vert_elem * q;
        Matrix3x3 Jx = Matrix3x3(0.f);
        for (int j = 0; j < nb_vert_elem; ++j) {
            int t = cb_topology[vid + j];
            Vector3 p = cb_p[t];
            Jx += glm::outerProduct(p, cb_dN[q_off + j]);
        }
        Matrix3x3 F = Jx * JXinv;
        Matrix3x3 P;
        scalar energy;
        if (material == 0) hooke_first(F, P, energy);
        else hooke_second(F, P, energy);
        P = P * glm::transpose(JXinv) * V;
        C += energy * V;

        for (int j = 0; j < nb_vert_elem; ++j) {
            grad_C[j] += P * cb_dN[q_off + j];
        }
    }

    C = (C < 0.f) ? -C : C; // abs
    if (C < 1e-12f) return;
    C = std::sqrt(C);

    // convert force to constraint gradient
    const scalar C_inv = 1.f / (2.f * C);
    for (int j = 0; j < nb_vert_elem; ++j) {
        grad_C[j] *= C_inv;
    }
    scalar sum_norm_grad = 0.f;
    for (int i = 0; i < nb_vert_elem; ++i) {
        sum_norm_grad += glm::dot(grad_C[i], grad_C[i]) * inv_mass[cb_topology[vid + i]];
    }
    if (sum_norm_grad < 1e-12f) return;
    const scalar alpha = 1.f / (stiffness * dt * dt);
    const scalar dt_lambda = -C / (sum_norm_grad + alpha);
    for (int i = 0; i < nb_vert_elem; ++i) {
        cb_p[cb_topology[vid + i]] += dt_lambda * inv_mass[cb_topology[vid + i]] * grad_C[i];
    }
}


void GPU_PBD_FEM::step(const scalar dt) const {
    int sub_it = 50;
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    scalar sub_dt = dt / sub_it;
    scalar young = 1e7f;
    scalar poisson = 0.35f;
    scalar lambda = young * poisson / ((1.f + poisson) * (1.f - 2.f * poisson));
    scalar mu = young / (2.f * (1.f + poisson));

    for (int i = 0; i < sub_it; ++i) {
        kernel_step_solver<<<(cb_position->nb + 255) / 256, 256>>>(cb_position->nb, sub_dt, Dynamic::gravity(),
                                                                   cb_position->buffer, cb_prev_position->buffer,
                                                                   cb_velocity->buffer, cb_forces->buffer,
                                                                   cb_inv_mass->buffer);

        for (int j = 0; j < c_offsets.size(); ++j) {
            kernel_constraint_solve_v0<<<c_nb_elem[j],nb_quadrature>>>(c_nb_elem[j], nb_quadrature, elem_nb_vert, sub_dt,
                                                          0, lambda,
                                                          c_offsets[j],
                                                          cb_dN->buffer,
                                                          cb_position->buffer, cb_topology->buffer,
                                                          cb_inv_mass->buffer,
                                                          cb_V->buffer, cb_JX_inv->buffer);

            kernel_constraint_solve_v0<<<c_nb_elem[j],nb_quadrature>>>(c_nb_elem[j], nb_quadrature, elem_nb_vert, sub_dt,
                                                          1, 2.f*mu,
                                                          c_offsets[j],
                                                          cb_dN->buffer,
                                                          cb_position->buffer, cb_topology->buffer,
                                                          cb_inv_mass->buffer,
                                                          cb_V->buffer, cb_JX_inv->buffer);

        }
        kernel_constraint_plane<<<(cb_position->nb + 255) / 256, 256>>>(cb_position->nb, Unit3D::right() * 0.01f,
                                                                        -Unit3D::right(),
                                                                        cb_position->buffer, cb_init_position->buffer);

        kernel_velocity_update<<<(cb_position->nb + 255) / 256, 256>>>(cb_position->nb, sub_dt,
                                                                       cb_position->buffer, cb_prev_position->buffer,
                                                                       cb_velocity->buffer);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    DebugUI::Begin("XPBD");
    DebugUI::Plot("Time GPU XPBD ", time);
    DebugUI::Value("Time", time);
    DebugUI::Range("Range", time);
    DebugUI::End();
}
