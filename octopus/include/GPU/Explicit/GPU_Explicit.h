#pragma once
#include <set>

#include "GPU/GPU_FEM.h"
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"
#include "GPU/VBD/GPU_VBD_FEM.h"

struct GPU_Explicit final : GPU_ParticleSystem {
    GPU_Explicit(const std::vector<Vector3>& positions, const std::vector<scalar>& masses, const int sub_it, const scalar damping)
    : GPU_ParticleSystem(positions, masses, new GPU_SemiExplicit(), sub_it), _damping(damping) { }

    void step(scalar dt) override;

    scalar _damping;

protected:
    ~GPU_Explicit() override;
};


struct GPU_Explicit_FEM final : GPU_FEM
{
    GPU_Explicit_FEM(Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
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

__global__ void kernel_explicit_fem_eval_force(
    int n, scalar damping,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
);
