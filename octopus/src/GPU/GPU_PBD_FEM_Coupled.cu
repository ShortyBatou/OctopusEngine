#include "GPU/GPU_PBD_FEM_Coupled.h"
#include <GPU/CUMatrix.h>

__device__ void xpbd_solve_coupled(const int nb_vert_elem, const scalar stiffness1, const scalar stiffness2, const scalar dt, const scalar* C, const Vector3* grad_C, scalar* inv_mass, int* topology, Vector3* p, int* mask)
{
    const scalar a1 = 1.f / (stiffness1 * dt * dt);
    const scalar a2 = 1.f / (stiffness2 * dt * dt);
    Matrix2x2 A(a1,0,0,a2);
    for (int i = 0; i < nb_vert_elem; ++i) {
        const scalar wi = inv_mass[topology[i]];
        A[0][0] += glm::dot(grad_C[i], grad_C[i]) * wi;
        A[1][0] += glm::dot(grad_C[i+nb_vert_elem], grad_C[i]) * wi;
        A[1][1] += glm::dot(grad_C[i+nb_vert_elem], grad_C[i+nb_vert_elem]) * wi;
    }

    A[0][1] = A[1][0]; // 2x2 symmetric matrix
    Vector2 dt_lambda = -glm::inverse(A) * Vector2(C[0], C[1]);
    for (int i = 0; i < nb_vert_elem; ++i) {
        const int vid = topology[i];
        if(mask[vid] == 1)
            p[vid] += (dt_lambda[0] * grad_C[i] + dt_lambda[1] * grad_C[i+nb_vert_elem]) * inv_mass[vid];
    }
}


__device__ void xpbd_constraint_fem_eval_coupled(
    const int nb_vert_elem, const scalar lambda, const scalar mu, const Material material, const Matrix3x3& Jx_inv, const scalar& V,
    Vector3* dN, Vector3* p, int* topology, scalar* C, Vector3* grad_C)
{
    const Matrix3x3 Jx = compute_transform(nb_vert_elem, p, topology, dN);
    const Matrix3x3 F = Jx * Jx_inv;

    Matrix3x3 P; scalar energy;

    eval_material(material, 0, lambda, mu, F, P, energy);
    P = P * glm::transpose(Jx_inv) * V;
    C[0] += energy * V;
    for (int i = 0; i < nb_vert_elem; ++i) {
        grad_C[i] += P * dN[i];
    }

    eval_material(material, 1, lambda, mu, F, P, energy);
    P = P * glm::transpose(Jx_inv) * V;
    C[1] += energy * V;
    for (int i = 0; i < nb_vert_elem; ++i) {
        grad_C[nb_vert_elem + i] += P * dN[i];
    }
}


__global__ void kernel_XPBD_Coupled_V0(
    const int n, const int nb_quadrature, const int nb_vert_elem, const scalar dt, // some global data
    const scalar stiffness_1, const scalar stiffness_2, const Material material, // material
    const int offset, // coloration
    Vector3* cb_dN,
    Vector3 *cb_p, int *cb_topology, // mesh
    scalar *inv_mass, int* mask,
    scalar *cb_V, Matrix3x3 *cb_JX_inv // element data (Volume * Weight, Inverse of initial jacobian)
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    const int eid = tid + offset / nb_vert_elem;
    const int vid = tid * nb_vert_elem + offset; // first vertice id in topology
    const int qid = eid * nb_quadrature;
    int* topology = cb_topology+vid;


    Vector3 grad_C[64];
    scalar C[2] = {0.f, 0.f};

    for (int j = 0; j < nb_vert_elem * 2; ++j)
        grad_C[j] = Vector3(0, 0, 0);

    for (int q = 0; q < nb_quadrature; ++q) { // must be possible to do in parrallel
        Vector3* dN = cb_dN + q * nb_vert_elem;
        xpbd_constraint_fem_eval_coupled(nb_vert_elem, stiffness_1, stiffness_2, material, cb_JX_inv[qid + q], cb_V[qid + q], dN, cb_p, topology, C, grad_C);
    }
    xpbd_convert_to_constraint(nb_vert_elem, C[0], grad_C);
    xpbd_convert_to_constraint(nb_vert_elem, C[1], grad_C + nb_vert_elem);
    xpbd_solve_coupled(nb_vert_elem, stiffness_1, stiffness_2, dt, C, grad_C, inv_mass, topology, cb_p, mask);
}


__device__ void xpbd_constraint_fem_eval_coupled_V2(const Material material, const scalar lambda, const scalar mu, const Matrix3x3& Jx_inv, const Matrix3x3& F, const scalar& V, scalar* C, Matrix3x3* P)
{
    for(int m = 0; m < 2; ++m) {
        eval_material(material, m, lambda, mu, F, P[m], C[m]);
        P[m] = P[m] * glm::transpose(Jx_inv) * V;
        C[m] *= V;
    }
}

__global__ void kernel_XPBD_Coupled_V2(
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
    const int eid = tid/nb_quadrature + offset / nb_vert_elem;
    const int vid = tid/nb_quadrature * nb_vert_elem + offset; // first vertice id in topology
    const int qid = eid * nb_quadrature;
    const int q = tid % nb_quadrature;
    int* topology = cb_topology+vid;

    __shared__ Matrix3x3 P[8];
    scalar C[8];

    for (int j = 0; j < nb_quadrature * 2; ++j) {
        P[j] = Matrix3x3(0.f);
        C[j] = 0.f;
    }

    Vector3* dN = cb_dN + q * nb_vert_elem;
    const Matrix3x3 F = compute_transform(nb_vert_elem, cb_p, topology, dN) * cb_JX_inv[qid + q];
    xpbd_constraint_fem_eval_coupled_V2(material, stiffness_1, stiffness_2, cb_JX_inv[qid + q], F, cb_V[qid + q], C+q*2, P+q*2);
    

}



void GPU_PBD_FEM_Coupled::step(const GPU_ParticleSystem* ps, const scalar dt) {
    for (int j = 0; j < c_offsets.size(); ++j) {
        kernel_XPBD_Coupled_V0<<<c_nb_elem[j],nb_quadrature>>>(c_nb_elem[j], nb_quadrature, elem_nb_vert, dt,
                                                       lambda, mu, _material,
                                                      c_offsets[j],
                                                      cb_dN->buffer,
                                                      ps->cb_position->buffer, cb_topology->buffer,
                                                      ps->cb_inv_mass->buffer, ps->cb_mask->buffer,
                                                      cb_V->buffer, cb_JX_inv->buffer);


    }
}
