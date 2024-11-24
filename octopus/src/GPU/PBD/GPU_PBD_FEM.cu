#include "GPU/PBD/GPU_PBD_FEM.h"
#include <Manager/Debug.h>
#include <GPU/PBD/GPU_PBD_FEM_Materials.h>
#include <GPU/CUMatrix.h>
#include <GPU/GPU_FEM.h>


__device__ void xpbd_solve(const int nb_vert_elem, const scalar stiffness, const scalar dt, const scalar& C, const Vector3* grad_C, scalar* inv_mass, int* topology, Vector3* p, int* mask)
{
    scalar sum_norm_grad = 0.f;
    for (int i = 0; i < nb_vert_elem; ++i) {
        sum_norm_grad += glm::dot(grad_C[i], grad_C[i]) * inv_mass[topology[i]];
    }
    if(sum_norm_grad < 1e-12) return;
    const scalar alpha = 1.f / (stiffness * dt * dt);
    const scalar dt_lambda = -C / (sum_norm_grad + alpha);
    for (int i = 0; i < nb_vert_elem; ++i) {
        int vid = topology[i];
        if(mask[vid] == 1) p[vid] += dt_lambda * inv_mass[vid] * grad_C[i];
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
    const scalar C_inv = C < 1e-5 ? 0.f : 1.f / (2.f * C);
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

        xpbd_solve(nb_vert_elem,(m==0)?stiffness_1:stiffness_2, dt, C, grad_C, inv_mass, topology, cb_p, mask);
    }
}

void GPU_PBD_FEM::step(const GPU_ParticleSystem* ps, const scalar dt) {
    for (int j = 0; j < c_offsets.size(); ++j) {
        kernel_XPBD_V0<<<c_nb_elem[j],nb_quadrature>>>(c_nb_elem[j], nb_quadrature, elem_nb_vert, dt,
                                                       lambda, mu, _material,
                                                      c_offsets[j],
                                                      cb_dN->buffer,
                                                      ps->buffer_position(), cb_topology->buffer,
                                                      ps->buffer_inv_mass(), ps->buffer_mask(),
                                                      cb_V->buffer, cb_JX_inv->buffer);


    }
}
