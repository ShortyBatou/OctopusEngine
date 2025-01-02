#pragma once
#include "Core/Base.h"
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Mesh/Mesh.h>
#include "Mesh/Elements.h"

#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Dynamic.h"
#include "GPU/VBD/GPU_VBD_FEM.h"

struct GPU_MG_Interpolation_Parameters
{
    int nb_vert_primitives;
    scalar weight;
    int* ids;
    int* primitives;
};


struct GPU_MG_Interpolation
{
    int nb_vert_primitives;
    scalar weight;
    Cuda_Buffer<int>* cb_ids; // vertices that will be interpolated
    Cuda_Buffer<int>* cb_primitives; // vertices that will be used for interpolation (lines, triangles, quads, etc.)
    ~GPU_MG_Interpolation()
    {
        delete cb_ids;
        delete cb_primitives;
    }
};


struct GPU_MG_VBD_FEM final : public GPU_VBD_FEM
{
    GPU_MG_VBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry,
                   const Material& material, const scalar& young, const scalar& poisson,
                   const scalar& damping,
                   const scalar& linear, const int& nb_iteration);

    void step(GPU_ParticleSystem* ps, scalar dt) override;

    [[nodiscard]] GPU_MG_Interpolation_Parameters get_interpolation_parameters(const int& i) const
    {
        GPU_MG_Interpolation_Parameters params{};
        params.nb_vert_primitives = interpolations[i]->nb_vert_primitives;
        params.weight = interpolations[i]->weight;
        params.ids = interpolations[i]->cb_ids->buffer;
        params.primitives = interpolations[i]->cb_primitives->buffer;
        return params;
    }

    std::vector<int> nb_iterations;
    int it_count;
    int level;

    std::vector<GPU_MG_Interpolation*> interpolations;
    std::vector<Thread_Data*> l_threads;
    std::vector<GPU_FEM_Data*> l_fems;
};


