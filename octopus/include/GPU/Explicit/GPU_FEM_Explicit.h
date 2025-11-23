#pragma once
#include <set>
#include "Core/Base.h"
#include "GPU/GPU_FEM.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"
#include "GPU/VBD/GPU_VBD_FEM.h"

struct GPU_FEM_Explicit final : GPU_FEM
{
    GPU_FEM_Explicit(Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                        scalar young, scalar poisson, Material material, scalar damping);

    void step(GPU_ParticleSystem* ps, scalar dt) override;

    void build_graph_color(const Mesh::Topology &topology, int nb_vertices);

    [[nodiscard]] GPU_Owners_Parameters get_owners_parameters() const
    {
        GPU_Owners_Parameters params{};
        params.offset = d_owners->cb_offset->buffer;
        params.nb = d_owners->cb_nb->buffer;
        params.eids = d_owners->cb_eids->buffer;
        params.ref_vid = d_owners->cb_ref_vid->buffer;
        return params;
    }

    scalar _damping;
    GPU_Owners_Data* d_owners;
};

__global__ void kernel_fem_eval_force(
    int n, scalar damping,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
);
