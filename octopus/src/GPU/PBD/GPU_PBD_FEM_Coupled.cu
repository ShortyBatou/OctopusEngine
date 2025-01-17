#include <GPU/CUMatrix.h>
#include "GPU/PBD/GPU_PBD_FEM_Coupled.h"
#include <GPU/PBD/GPU_PBD_FEM_Materials.h>

__device__ void xpbd_solve_coupled(const int nb_vert_elem, const scalar dt,
                                   Material_Data mt, GPU_ParticleSystem_Parameters ps,
                                   const int* topology, const scalar* C, const Vector3* grad_C)
{
    const scalar a1 = 1.f / (mt.lambda * dt * dt);
    const scalar a2 = 1.f / (mt.mu * dt * dt);
    Matrix2x2 A(a1, 0, 0, a2);
    for (int i = 0; i < nb_vert_elem; ++i)
    {
        const scalar wi = ps.w[topology[i]];
        A[0][0] += glm::dot(grad_C[i], grad_C[i]) * wi;
        A[1][0] += glm::dot(grad_C[i + nb_vert_elem], grad_C[i]) * wi;
        A[1][1] += glm::dot(grad_C[i + nb_vert_elem], grad_C[i + nb_vert_elem]) * wi;
    }

    A[0][1] = A[1][0]; // 2x2 symmetric matrix
    Vector2 dt_lambda = -glm::inverse(A) * Vector2(C[0], C[1]);
    for (int i = 0; i < nb_vert_elem; ++i)
    {
        const int vid = topology[i];
        if (ps.mask[vid] == 1)
            ps.p[vid] += (dt_lambda[0] * grad_C[i] + dt_lambda[1] * grad_C[i + nb_vert_elem]) * ps.w[vid];
    }
}

__device__ void xpbd_constraint_fem_eval_coupled(
    const int nb_vert_elem, Material_Data mt, const int* topology, const Matrix3x3& Jx_inv, const scalar& V,
    const Vector3* dN, const Vector3* p, scalar* C, Vector3* grad_C)
{
    const Matrix3x3 Jx = compute_transform(nb_vert_elem, p, topology, dN);
    const Matrix3x3 F = Jx * Jx_inv;

    Matrix3x3 P;
    scalar energy;

    eval_material(mt.material, 0, mt.lambda, mt.mu, F, P, energy);
    P = P * glm::transpose(Jx_inv) * V;
    C[0] += energy * V;
    for (int i = 0; i < nb_vert_elem; ++i)
    {
        grad_C[i] += P * dN[i];
    }

    eval_material(mt.material, 1, mt.lambda, mt.mu, F, P, energy);
    P = P * glm::transpose(Jx_inv) * V;
    C[1] += energy * V;
    for (int i = 0; i < nb_vert_elem; ++i)
    {
        grad_C[nb_vert_elem + i] += P * dN[i];
    }
}

__global__ void kernel_XPBD_Coupled_V0(
    const int n, const int offset, const scalar dt, const int* eids, Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    const int eid = eids[tid + offset];
    const int vid = eid * fem.elem_nb_vert; // first vertice id in topology
    const int qid = eid * fem.nb_quadrature;
    const int* topology = fem.topology + vid;


    Vector3 grad_C[64];
    scalar C[2] = {0.f, 0.f};

    for (int j = 0; j < fem.elem_nb_vert * 2; ++j)
        grad_C[j] = Vector3(0, 0, 0);

    for (int q = 0; q < fem.nb_quadrature; ++q)
    {
        // must be possible to do in parrallel
        const Vector3* dN = fem.dN + q * fem.elem_nb_vert;
        xpbd_constraint_fem_eval_coupled(fem.elem_nb_vert, mt, topology, fem.JX_inv[qid + q], fem.V[qid + q], dN, ps.p,
                                         C, grad_C);
    }
    xpbd_convert_to_constraint(fem.elem_nb_vert, C[0], grad_C);
    xpbd_convert_to_constraint(fem.elem_nb_vert, C[1], grad_C + fem.elem_nb_vert);
    xpbd_solve_coupled(fem.elem_nb_vert, dt, mt, ps, topology, C, grad_C);
}


GPU_PBD_FEM_Coupled::GPU_PBD_FEM_Coupled(
    const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology,
    const scalar young, const scalar poisson, const Material material)
        : GPU_PBD_FEM(element, geometry, topology, young, poisson, material) { }


void GPU_PBD_FEM_Coupled::step(GPU_ParticleSystem* ps, const scalar dt)
{
    for (int j = 0; j < d_thread->nb_kernel; ++j)
    {
        kernel_XPBD_Coupled_V0<<<d_thread->grid_size[j],d_thread->block_size[j]>>>(
            d_thread->nb_threads[j], d_thread->offsets[j], dt, cb_eid->buffer,
            *d_material, ps->get_parameters(), get_fem_parameters());
    }
}
