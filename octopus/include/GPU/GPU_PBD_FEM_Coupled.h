#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "cuda_runtime.h"
#include "GPU/GPU_PBD_FEM.h"

struct GPU_PBD_FEM_Coupled final : GPU_PBD_FEM {
    GPU_PBD_FEM_Coupled(Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, scalar young, scalar poisson, Material material);

    ~GPU_PBD_FEM_Coupled() override = default;
    void step(const GPU_ParticleSystem *ps, scalar dt) override;
};

//__device__ void xpbd_solve_coupled(int nb_vert_elem, scalar stiffness1, scalar stiffness2, scalar dt, scalar* C, Vector3* grad_C, scalar* inv_mass, int* topology, Vector3* p);

__device__ void xpbd_constraint_fem_eval_coupled(
    int nb_vert_elem, const Matrix3x3& Jx_inv, const scalar& V,
    Vector3* dN, Vector3* p, int* topology, scalar* C, Vector3* grad_C);

__global__ void kernel_XPBD_Coupled_V0(
    int n, int nb_quadrature, int nb_vert_elem, scalar dt, // some global data
    scalar stiffness_1, scalar stiffness_2, // material
    int offset, // coloration
    Vector3* cb_dN,
    Vector3 *cb_p, int *cb_topology, // mesh
    scalar *inv_mass,
    scalar *cb_V, Matrix3x3 *cb_JX_inv // element data (Volume * Weight, Inverse of initial jacobian)
);