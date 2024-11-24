#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Mesh/Mesh.h>

#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_Plane_Fix final : GPU_Dynamic {
    GPU_Plane_Fix(const std::vector<Vector3>& positions, const Vector3& o, const Vector3& n);
    Vector3 com, offset, origin, normal;
    Matrix3x3 rot;
    void step(const GPU_ParticleSystem *ps, scalar dt) override;
};

struct GPU_FEM : GPU_Dynamic {
    GPU_FEM(Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                         scalar young, scalar poisson, Material material);

    virtual void build_fem_const(const Mesh::Geometry& geometry, const Mesh::Topology& topology);

    FEM_Shape* shape;
    int elem_nb_vert;  // nb vertice per element
    int nb_quadrature; // nb quadrature for fem
    int nb_element;    // element total

    scalar lambda;
    scalar mu;

    std::vector<int> c_nb_elem; // nb element for a color
    std::vector<int> c_offsets; // topo index offset for each color (topology is sorted by color)

    Cuda_Buffer<int> *cb_topology;
    Cuda_Buffer<Matrix3x3> *cb_JX_inv;
    Cuda_Buffer<scalar> *cb_V;
    Cuda_Buffer<Vector3> *cb_dN;
    Material _material;
};

__global__ void kernel_constraint_plane(int n, float dt, Vector3 origin, Vector3 normal, Vector3 *p, Vector3 *v,
                                        Vector3 *f);

__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3 *pos, int *topology, Vector3 *dN);