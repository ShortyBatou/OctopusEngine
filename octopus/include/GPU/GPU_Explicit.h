#pragma once
#include <set>

#include "GPU_FEM.h"
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_Explicit : GPU_ParticleSystem {
    GPU_Explicit(const std::vector<Vector3>& positions, const std::vector<scalar>& masses, const int sub_it, const scalar damping)
    : GPU_ParticleSystem(positions, masses, new GPU_SemiExplicit(), sub_it), _damping(damping) { }

    void step(scalar dt) const;

    scalar _damping;

protected:
    ~GPU_Explicit() override;
};


struct GPU_Explicit_FEM final : GPU_FEM
{
    GPU_Explicit_FEM(Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                        scalar young, scalar poisson, Material material, scalar damping);
    void step(const GPU_ParticleSystem* ps, scalar dt) override;

    void build_graph_color(const Mesh::Topology &topology, int nb_vertices);
    scalar _damping;
    int _block_size;
    int _nb_threads;

    Cuda_Buffer<int> *cb_neighbors_offset; // offset to the first neighbors for each vertice
    Cuda_Buffer<int> *cb_nb_neighbors; // number of neighbors for each vertices
    Cuda_Buffer<int> *cb_neighbors; // element's id that own a vertices
    Cuda_Buffer<int> *cb_ref_vid; // local vertice id in each neighbors element
};