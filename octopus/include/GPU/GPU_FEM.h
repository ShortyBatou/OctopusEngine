#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "cuda_runtime.h"
#include "GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_Plane_Fix final : GPU_Dynamic {
    GPU_Plane_Fix(const std::vector<Vector3>& positions, const Vector3& o, const Vector3& n);
    Vector3 com, offset, origin, normal;
    Matrix3x3 rot;
    void step(const GPU_ParticleSystem *ps, scalar dt) override;
};

struct GPU_PBD_FEM : GPU_Dynamic {
    FEM_Shape* shape;
    int elem_nb_vert;  // nb vertice per element
    int nb_quadrature; // nb quadrature for fem
    int nb_color;      // color total
    int nb_element;    // element total

    scalar lambda;
    scalar mu;

    std::vector<int> c_nb_elem; // nb element for a color
    std::vector<int> c_offsets; // topo index offset for each color (topology is sorted by color)

    Cuda_Buffer<int> *cb_topology;
    Cuda_Buffer<Matrix3x3> *cb_JX_inv;
    Cuda_Buffer<scalar> *cb_V;
    Cuda_Buffer<Vector3> *cb_dN;

    std::vector<int> colors; // mesh coloration (used for debug)

    GPU_PBD_FEM(Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology,
                scalar young, scalar poisson);

    ~GPU_PBD_FEM() override = default;

    void step(const GPU_ParticleSystem *ps, scalar dt) override;
private:
    int build_graph_color(const Mesh::Topology& topology, int nb_vert, std::vector<int>& colors);
    std::vector<int> build_topology_by_color(const std::vector<int>& colors, const std::vector<int>& topology);
    void build_fem_const(const Mesh::Geometry& geometry, const Mesh::Topology& topology);
};

__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3 *pos, Vector3 *dN);

__global__ void kernel_constraint_solve(int n, int nb_quadrature, int nb_vert_elem, int offset, Vector3 *p,
                                        int *topology, Vector3 *dN, scalar *V, Matrix3x3 *JX_inv);

__global__ void kernel_constraint_plane(int n, float dt, Vector3 origin, Vector3 normal, Vector3 *p, Vector3 *v,
                                        Vector3 *f);
