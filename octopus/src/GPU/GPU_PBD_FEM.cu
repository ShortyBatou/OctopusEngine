#include "GPU/GPU_PBD_FEM.h"
#include <Manager/Debug.h>
#include <GPU/CUMatrix.h>

// fem global function
__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3 *pos, int *topology, Vector3 *dN) {
    // Compute transform (reference => scene)
    Matrix3x3 Jx = Matrix3x3(0.f);
    for (int j = 0; j < nb_vert_elem; ++j) {
        Jx += glm::outerProduct(pos[topology[j]], dN[j]);
    }
    return Jx;
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
    P = 4.f * F * E;
    C = 2.f * squared_trace(E);
}

__device__ void hooke_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const float trace = mat3x3_trace(0.5f * (glm::transpose(F) + F) - Matrix3x3(1.f));
    // P(F) = 2E   C(F) = tr(E)�
    C = trace * trace;
    P = 2.f * trace * Matrix3x3(1.f);
}

__device__ void hooke_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const Matrix3x3 E = 0.5f * (glm::transpose(F) + F) - Matrix3x3(1.f);
    // P(F) = 2E   C(F) = tr(E)�
    C = 2.f * squared_trace(E);
    P = 4.f * E;
}


__device__ void snh_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C, scalar alpha) {
    const scalar I_3 = glm::determinant(F);
    const scalar detF = I_3 - alpha;
    C = (detF) * (detF);
    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * detF * d_detF;
}

__device__ void snh_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    C = squared_norm(F) - 3.f;
    P = 2.f * F;
}

__device__ void dsnh_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    scalar I_3 = glm::determinant(F);
    scalar detF = I_3 - 1;
    C = (detF) * (detF);
    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * detF * d_detF;
}

__device__ void dsnh_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    scalar I_3 = glm::determinant(F);
    C = squared_norm(F) - 3.f - 2.f * (I_3 - 1.f);

    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * F - 2.f * d_detF;
}

__device__ void eval_material(const Material material, const int m, const scalar lambda, const scalar mu, const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    switch (material) {
        case Hooke :
            if (m == 0) hooke_first(F, P, energy);
            else hooke_second(F, P, energy); break;
        case StVK :
            if (m == 0) stvk_first(F, P, energy);
            else stvk_second(F, P, energy); break;
        case NeoHooke :
            if (m == 0) snh_first(F, P, energy, 1.f + mu / lambda);
            else snh_second(F, P, energy); break;
        case Stable_NeoHooke :
            if (m == 0) dsnh_first(F, P, energy);
            else dsnh_second(F, P, energy); break;
    }
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

__device__ void xpbd_constraint_fem_eval(const Material material, const int m, const scalar lambda, const scalar mu, const int nb_vert_elem, const Matrix3x3& Jx_inv, const scalar& V, Vector3* dN, Vector3* p, int* topology, scalar& C, Vector3* grad_C)
{

    const Matrix3x3 Jx = compute_transform(nb_vert_elem, p, topology, dN);
    const Matrix3x3 F = Jx * Jx_inv;

    Matrix3x3 P;
    scalar energy;

    eval_material(material, m, lambda, mu, F, P, energy);
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
    const scalar C_inv = C < 1e-12 ? 0.f : 1.f / (2.f * C);
    for (int i = 0; i < nb_vert_elem; ++i) {
        grad_C[i] *= C_inv;
    }
}

__global__ void kernel_XPBD_V0(
    const int n, const int nb_quadrature, const int nb_vert_elem, const scalar dt, // some global data
    const scalar stiffness_1, const scalar stiffness_2, const Material material, // material
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
    for(int m = 0; m < 2; ++m) // nb materials
    {
        Vector3 grad_C[32];
        scalar C = 0.f;
        for (int j = 0; j < nb_vert_elem; ++j)
            grad_C[j] = Vector3(0, 0, 0);

        for (int q = 0; q < nb_quadrature; ++q) { // must be possible to do in parrallel
            Matrix3x3 JX_inv = cb_JX_inv[qid + q];
            scalar V = cb_V[qid + q];
            Vector3* dN = cb_dN + q * nb_vert_elem;
            xpbd_constraint_fem_eval(material, m, stiffness_1, stiffness_2, nb_vert_elem, JX_inv, V, dN, cb_p, topology, C, grad_C);
        }

        xpbd_convert_to_constraint(nb_vert_elem, C, grad_C);
        if(C < 1e-12f) continue;

        xpbd_solve(nb_vert_elem,(m==0)?stiffness_1:stiffness_2, dt, C, grad_C, inv_mass, topology, cb_p);
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
                                                       lambda, mu, _material,
                                                      c_offsets[j],
                                                      cb_dN->buffer,
                                                      ps->cb_position->buffer, cb_topology->buffer,
                                                      ps->cb_inv_mass->buffer,
                                                      cb_V->buffer, cb_JX_inv->buffer);


    }
}
