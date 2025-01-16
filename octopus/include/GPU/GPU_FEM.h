#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Mesh/Mesh.h>

#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"

struct Material_Data
{
    Material material;
    scalar lambda, mu;
};

struct Thread_Data
{
    int nb_kernel;
    std::vector<int> nb_threads;
    std::vector<int> block_size;
    std::vector<int> grid_size;
    std::vector<int> offsets; // offset to the first data
};

struct GPU_FEM_Data
{
    int elem_nb_vert;                       // S = nb vertice per element
    int nb_quadrature;                      // Q = nb quadrature for fem
    int nb_element;                         // E = element total
    Cuda_Buffer<int> *cb_topology;          // E * S = mesh topology
    Cuda_Buffer<Matrix3x3> *cb_JX_inv;      // E * Q = Init inverse jacobian
    Cuda_Buffer<scalar> *cb_V;              // E * Q = Init volume
    Cuda_Buffer<scalar> *cb_weights;        // Q = quadrature weights
    Cuda_Buffer<Vector3> *cb_dN;            // Q * S = shape functions derivatives

    ~GPU_FEM_Data()
    {
        delete cb_topology;
        delete cb_JX_inv;
        delete cb_V;
        delete cb_weights;
        delete cb_dN;
    }
};

struct GPU_FEM_Pameters
{
    int nb_element;
    int elem_nb_vert;
    int nb_quadrature;
    int* topology;
    Matrix3x3* JX_inv;
    scalar* V;
    scalar* weights;
    Vector3* dN;
};

struct GPU_Plane_Fix final : GPU_Dynamic {
    GPU_Plane_Fix(const std::vector<Vector3>& positions, const Vector3& o, const Vector3& n);
    Vector3 com, offset, origin, normal;
    bool all;
    Matrix3x3 rot;
    void set_for_all(bool state) { all = state; }
    void step(GPU_ParticleSystem *ps, scalar dt) override;
};

struct GPU_FEM : GPU_Dynamic {
    GPU_FEM(Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                         scalar young, scalar poisson, Material material);

    virtual GPU_FEM_Data* build_fem_const(const Element& element, const Mesh::Geometry& geometry, const Mesh::Topology& topology);
    std::vector<scalar> get_stress(const GPU_ParticleSystem *ps) const;
    std::vector<scalar> get_volume(const GPU_ParticleSystem *ps) const;
    std::vector<scalar> get_volume_diff(const GPU_ParticleSystem *ps) const;
    std::vector<Vector3> get_residual(const GPU_ParticleSystem *ps, scalar dt) const;

    [[nodiscard]] GPU_FEM_Pameters get_fem_parameters() const {
        GPU_FEM_Pameters param{};
        param.nb_element = d_fem->nb_element;
        param.elem_nb_vert = d_fem->elem_nb_vert;
        param.nb_quadrature = d_fem->nb_quadrature;
        param.topology = d_fem->cb_topology->buffer;
        param.JX_inv = d_fem->cb_JX_inv->buffer;
        param.V = d_fem->cb_V->buffer;
        param.weights = d_fem->cb_weights->buffer;
        param.dN = d_fem->cb_dN->buffer;
        return param;
    }

    ~GPU_FEM() override
    {
        delete d_material;
        delete d_thread;
        delete d_fem;
    }

    // MATERIAL DATA
    Material_Data* d_material;
    // Multi-threading data (coloration)
    Thread_Data* d_thread;
    // FEM DATA
    GPU_FEM_Data* d_fem;

private:
    Cuda_Buffer<scalar> *cb_elem_data;
};

__global__ void kernel_constraint_plane(int n, float dt, Vector3 origin, Vector3 normal, Vector3 *p, Vector3 *v,
                                        Vector3 *f);

__device__ Matrix3x3 compute_transform(int nb_vert_elem, const Vector3 *pos, const int *topology, const Vector3 *dN);