#include "GPU/GPU_PBD.h"

#include <Manager/Debug.h>
#include <Manager/Key.h>

// global device function
__device__ scalar mat3x3_trace(const Matrix3x3 &m) {
    return m[0][0] + m[1][1] + m[2][2];
}

__device__ scalar squared_trace(const Matrix3x3 &m)
{
    return m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2] + (m[0][1] * m[1][0] + m[1][2] * m[2][1] + m[2][0] * m[0][2]) * 2.f;
}

__device__ void print_vec(const Vector3 &v) {
    printf("x:%f y:%f z:%f", v.x, v.y, v.z);
}

__device__ void print_mat(const Matrix3x3 &m) {
    printf("%f %f %f %f %f %f %f %f %f", m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2],
           m[2][2]);
}
// fem global function

__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3 *pos, int *topology, Vector3 *dN) {
    // Compute transform (reference => scene)
    Matrix3x3 Jx = Matrix3x3(0.f);
    for (int j = 0; j < nb_vert_elem; ++j) {
        Jx += glm::outerProduct(pos[topology[j]], dN[j]);
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

__global__ void kernel_constraint_plane(const int n, const Vector3 origin, const Vector3 normal, const Vector3 com, const Vector3 offset, const Matrix3x3 rot, Vector3 *p, Vector3 *p_init) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = dot(p_init[i] - origin, normal);
    if (s > 0) {
        const Vector3 target = offset + com + rot * (p_init[i] - com);
        p[i] += (target - p[i]);
    }
}


__device__ void stvk_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const scalar trace = mat3x3_trace(0.5f * (glm::transpose(F) * F - Matrix3x3(1.f)));
    C = trace * trace;
    P = (2.f * trace * F);
}

__device__ void stvk_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const Matrix3x3 E = 0.5f * (glm::transpose(F) * F - Matrix3x3(1.f));
    P = 2.f * F * E;
    C = squared_trace(E);
}

__device__ void xpbd_solve(const int nb_vert_elem, const scalar stiffness, const scalar dt, const scalar& C, const Vector3* grad_C, scalar* inv_mass, int* topology, Vector3* p)
{
    scalar sum_norm_grad = 0.f;
    for (int i = 0; i < nb_vert_elem; ++i) {
        sum_norm_grad += glm::dot(grad_C[i], grad_C[i]) * inv_mass[topology[i]];
    }
    if(sum_norm_grad < 1e-12) return;
    const scalar alpha = 1.f / (stiffness * dt * dt);
    const scalar dt_lambda = -C / (sum_norm_grad + alpha);
    for (int i = 0; i < nb_vert_elem; ++i) {
        p[topology[i]] += dt_lambda * inv_mass[topology[i]] * grad_C[i];
    }
}

__device__ void xpbd_constraint_fem_eval(const int m, const int nb_vert_elem, const Matrix3x3& Jx_inv, const scalar& V, Vector3* dN, Vector3* p, int* topology, scalar& C, Vector3* grad_C)
{

    const Matrix3x3 Jx = compute_transform(nb_vert_elem, p, topology, dN);
    const Matrix3x3 F = Jx * Jx_inv;

    Matrix3x3 P;
    scalar energy;
    if (m == 0) stvk_first(F, P, energy);
    else stvk_second(F, P, energy);

    P = P * glm::transpose(Jx_inv) * V;
    C += energy * V;

    for (int i = 0; i < nb_vert_elem; ++i) {
        grad_C[i] += P * dN[i];
    }
}

__device__ void xpbd_convert_to_constraint(const int nb_vert_elem, scalar& C, Vector3* grad_C)
{
    // convert force to constraint gradient
    C = (C < 0.f) ? -C : C; // abs
    C = std::sqrt(C);
    if(C < 1e-12) return;
    const scalar C_inv = 1.f / (2.f * C);
    for (int i = 0; i < nb_vert_elem; ++i) {
        grad_C[i] *= C_inv;
    }
}

__global__ void kernel_XPBD_V0(
    const int n, const int nb_quadrature, const int nb_vert_elem, const scalar dt, // some global data
    const scalar stiffness_1, const scalar stiffness_2, // material
    const int offset, // coloration
    Vector3* cb_dN,
    Vector3 *cb_p, int *cb_topology, // mesh
    scalar *inv_mass,
    scalar *cb_V, Matrix3x3 *cb_JX_inv // element data (Volume * Weight, Inverse of initial jacobian)
)
{

    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    const int eid = tid + offset / nb_vert_elem;
    const int vid = tid * nb_vert_elem + offset; // first vertice id in topology
    const int qid = eid * nb_quadrature;
    int* topology = cb_topology+vid;
    for(int m = 0; m < 2; ++m)
    {
        Vector3 grad_C[32];
        scalar C = 0.f;
        for (int j = 0; j < nb_vert_elem; ++j)
            grad_C[j] = Vector3(0, 0, 0);

        for (int q = 0; q < nb_quadrature; ++q) {
            Matrix3x3 JX_inv = cb_JX_inv[qid + q];
            scalar V = cb_V[qid + q];
            Vector3* dN = cb_dN + q * nb_vert_elem;
            xpbd_constraint_fem_eval(m, nb_vert_elem, JX_inv, V, dN, cb_p, topology, C, grad_C);
        }

        xpbd_convert_to_constraint(nb_vert_elem, C, grad_C);
        if(C < 1e-12f) continue;

        xpbd_solve(nb_vert_elem,(m==0)?stiffness_1:stiffness_2, dt, C, grad_C, inv_mass, topology, cb_p);
    }
}

void GPU_PBD::step(const scalar dt) const {

    const scalar sub_dt = dt / static_cast<scalar>(iteration);

    for(int i = 0; i < iteration; ++i) {
        kernel_step_solver<<<(nb() + 255) / 256, 256>>>(nb(), sub_dt, Dynamic::gravity(),
                                                               cb_position->buffer, cb_prev_position->buffer,
                                                               cb_velocity->buffer, cb_forces->buffer,
                                                               cb_inv_mass->buffer);
        for(auto* c : dynamic) {
            c->step(this, sub_dt);
        }

        kernel_velocity_update<<<(cb_position->nb + 255) / 256, 256>>>(cb_position->nb, sub_dt,
                                                                   cb_position->buffer, cb_prev_position->buffer,cb_velocity->buffer);
    }

}

void GPU_Plane_Fix::step(const GPU_ParticleSystem *ps, const scalar dt) {
    kernel_constraint_plane<<<(ps->nb() + 255) / 256, 256>>>(
        ps->nb(), origin, normal, com, offset, rot,
        ps->cb_position->buffer, ps->cb_init_position->buffer);
}

void GPU_PBD_FEM::step(const GPU_ParticleSystem* ps, const scalar dt) {
    for (int j = 0; j < c_offsets.size(); ++j) {
        kernel_XPBD_V0<<<c_nb_elem[j],nb_quadrature>>>(c_nb_elem[j], nb_quadrature, elem_nb_vert, dt,
                                                       lambda, mu*2.f,
                                                      c_offsets[j],
                                                      cb_dN->buffer,
                                                      ps->cb_position->buffer, cb_topology->buffer,
                                                      ps->cb_inv_mass->buffer,
                                                      cb_V->buffer, cb_JX_inv->buffer);


    }
}
