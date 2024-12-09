#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Mesh/Mesh.h>
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_VBD_FEM final : GPU_Dynamic {
    GPU_VBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry, const Material& material,
        const scalar& young, const scalar& poisson, const scalar& damping);
    void step(const GPU_ParticleSystem* ps, scalar dt) override;
    void build_graph_color(const Mesh::Topology &topology, int nb_vertices,
        std::vector<int> &colors, std::vector<std::vector<int>>& e_neighbors, std::vector<std::vector<int>>& ref_id);
    void build_fem_const(const Mesh::Geometry &geometry, const Mesh::Topology& topology);
    void GPU_VBD_FEM::sort_by_color(int nb_vertices, const std::vector<std::vector<int>>& e_neighbors, const std::vector<std::vector<int>>& e_ref_id);
    FEM_Shape* shape;
    int elem_nb_vert;  // nb vertice per element
    int nb_quadrature; // nb quadrature for fem
    int nb_color;      // color total
    int nb_element;    // element total

    scalar _lambda;
    scalar _mu;
    Material _material;
    scalar _damping;

    std::vector<int> c_block_size;
    std::vector<int> c_nb_threads; // nb vertices for a color
    std::vector<int> c_offsets; // offset to the first vertices

    Cuda_Buffer<int> *cb_neighbors_offset; // offset to the first neighbors for each vertice
    Cuda_Buffer<int> *cb_nb_neighbors; // number of neighbors for each vertices
    Cuda_Buffer<int> *cb_neighbors; // element's id that own a vertices
    Cuda_Buffer<int> *cb_ref_vid; // local vertice id in each neighbors element
    Cuda_Buffer<Vector3>* y; // gets ot from VBD solver
    Cuda_Buffer<int> *cb_topology; // mesh topology
    Cuda_Buffer<Matrix3x3> *cb_JX_inv; // inverse of init state jacobian
    Cuda_Buffer<scalar> *cb_V; // quadrature
    Cuda_Buffer<Vector3> *cb_dN; // shape function derivatives

    std::vector<int> colors; // mesh coloration (used for debug)
};