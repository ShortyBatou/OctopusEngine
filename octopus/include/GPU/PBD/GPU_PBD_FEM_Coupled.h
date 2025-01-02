#pragma once

#include "Core/Base.h"
#include <Mesh/Mesh.h>
#include "Mesh/Elements.h"
#include "cuda_runtime.h"
#include <GPU/GPU_ParticleSystem.h>
#include "GPU/PBD/GPU_PBD_FEM.h"

struct GPU_PBD_FEM_Coupled final : GPU_PBD_FEM
{
    GPU_PBD_FEM_Coupled(Element element, const Mesh::Geometry& geometry, const Mesh::Topology& topology, scalar young,
                        scalar poisson, Material material);

    ~GPU_PBD_FEM_Coupled() override = default;
    void step(GPU_ParticleSystem* ps, scalar dt) override;
};

//__device__ void xpbd_solve_coupled(int nb_vert_elem, scalar stiffness1, scalar stiffness2, scalar dt, scalar* C, Vector3* grad_C, scalar* inv_mass, int* topology, Vector3* p);

__device__ void xpbd_constraint_fem_eval_coupled(
    int nb_vert_elem, Material_Data mt, const int* topology, const Matrix3x3& Jx_inv, const scalar& V,
    const Vector3* dN, const Vector3* p, scalar* C, Vector3* grad_C);

__global__ void kernel_XPBD_Coupled_V0(
    int n, int offset, scalar dt, const int* eids, Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem
);
