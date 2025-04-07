#pragma once
#include "Core/Base.h"
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Mesh/Mesh.h>
#include "Mesh/Elements.h"

#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Dynamic.h"
#include "GPU/VBD/GPU_VBD_FEM.h"

struct GPU_Adjacence_Parameters
{
    int* nb; // [N] the number of adjacent vertices per vertices
    int* offset; // [N] the offset per vertices to read the first value
    int* ids; // [N] the vertex id
    int* adj; // [Sum(nb)] adjacent vertices' id
    scalar* values; // [Sum(nb)] their corresponding weights
};

struct GPU_Adjacence
{
    GPU_Adjacence() = default;

    Cuda_Buffer<int>* cb_nb; // vertices that will be interpolated
    Cuda_Buffer<int>* cb_offset; // vertices that will be interpolated
    Cuda_Buffer<int>* cb_ids; // vertices that will be interpolated
    Cuda_Buffer<int>* cb_adj; // vertices that will be interpolated
    Cuda_Buffer<scalar>* cb_values; // vertices that will be interpolated

    ~GPU_Adjacence()
    {
        delete cb_nb;
        delete cb_offset;
        delete cb_ids;
        delete cb_adj;
        delete cb_values;
    }
};

struct GPU_MG_Interpolation_Parameters
{
    int nb_ids;
    int nb_vert_primitives;
    scalar weight;
    int* ids;
    int* primitives;
};


struct GPU_MG_Interpolation
{
    GPU_MG_Interpolation() = default;
    GPU_MG_Interpolation(const int nb_vert, const scalar w, const std::vector<int>& ids, const std::vector<int>& primitives)
    {
        nb_ids = static_cast<int>(ids.size());
        cb_ids = new Cuda_Buffer<int>(ids);
        cb_primitives = new Cuda_Buffer<int>(primitives);
        nb_vert_primitives = nb_vert;
        weight = w;
    }

    int nb_ids;
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
                   const scalar& linear, const int& nb_iteration,
                   const scalar& density, const Mass_Distribution& mass_distrib,
                   GPU_ParticleSystem* ps);

    void compute_intertia(GPU_ParticleSystem* ps, scalar dt) const;
    void step(GPU_ParticleSystem* ps, scalar dt) override;


    [[nodiscard]] GPU_Adjacence_Parameters get_prolongation_parameters(const int i) const
    {
        GPU_Adjacence_Parameters params{};
        params.nb = prolongations[i]->cb_nb->buffer;
        params.ids = prolongations[i]->cb_ids->buffer;
        params.offset = prolongations[i]->cb_offset->buffer;
        params.adj = prolongations[i]->cb_adj->buffer;
        params.values = prolongations[i]->cb_values->buffer;
        return params;
    }

    [[nodiscard]] GPU_Adjacence_Parameters get_restriction_parameters(const int i) const
    {
        GPU_Adjacence_Parameters params{};
        params.nb = restrictions[i]->cb_nb->buffer;
        params.ids = restrictions[i]->cb_ids->buffer;
        params.ids = restrictions[i]->cb_ids->buffer;
        params.offset = restrictions[i]->cb_offset->buffer;
        params.adj = restrictions[i]->cb_adj->buffer;
        params.values = restrictions[i]->cb_values->buffer;
        return params;
    }

     GPU_Adjacence* compute_interpolation_adj_matrix(
        const FEM_Shape* from_shape, const FEM_Shape* target_shape,
        const Mesh::Topology& topology, const Mesh::Geometry& geometry,
        const std::vector<scalar>& masses, scalar density);

    ~GPU_MG_VBD_FEM() override {
        for(int i = 0; i < prolongations.size(); ++i) {
            delete prolongations[i];
            delete restrictions[i];
        }

        for(int i = 0; i < l_threads.size(); ++i) {
            if(i == level) continue;
            delete l_threads[i];
            delete masses[i];
            delete interias[i];
            delete l_fems[i];
            delete l_owners[i];
        }

        delete cb_prev_state;
    }

    std::vector<int> nb_iterations;
    int it_count;
    int level;

    Cuda_Buffer<Vector3>* cb_prev_state;
    std::vector<Cuda_Buffer<Vector3>*> interias;
    std::vector<Cuda_Buffer<scalar>*> masses;
    std::vector<GPU_Adjacence*> prolongations;
    std::vector<GPU_Adjacence*> restrictions;
    std::vector<Thread_Data*> l_threads;
    std::vector<GPU_FEM_Data*> l_fems;
    std::vector<GPU_Owners_Data*> l_owners;
};


