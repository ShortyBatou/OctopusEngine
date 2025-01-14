#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>

#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "cuda_runtime.h"
#include "GPU/GPU_FEM.h"

struct GPU_PBD_FEM : GPU_FEM
{
    std::vector<int> colors; // mesh coloration (used for debug)

    GPU_PBD_FEM(Element element, const Mesh::Geometry& geometry, const Mesh::Topology& topology, scalar young,
                scalar poisson, Material material);

    ~GPU_PBD_FEM() override = default;

    void step(GPU_ParticleSystem* ps, scalar dt) override;

protected:
    Cuda_Buffer<int>* cb_eid;
    void build_graph_color(const Mesh::Topology& topology, int nb_vert, std::vector<int>& colors) const;
    void build_thread_by_color(const std::vector<int>& colors);
};

__device__ void xpbd_convert_to_constraint(int nb_vert_elem, scalar& C, Vector3* grad_C);
__global__ void kernel_constraint_solve(int n, int offset, scalar dt, const int* eids, Material_Data mt,
                                        GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem);
