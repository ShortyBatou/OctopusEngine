#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include <Dynamic/FEM/FEM_Shape.h>
#include "GPU/Cuda_Buffer.h"
#include <Manager/Dynamic.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct GPU_PBD_FEM {
    int nb_elem;
    int nb_color;
    int nb_quadrature;
    int nb_verts;
    // Particles
    Cuda_Buffer<Vector3>* cb_position;
    Cuda_Buffer<Vector3>* cb_prev_position;
    Cuda_Buffer<Vector3>* cb_velocity;
    Cuda_Buffer<Vector3>* cb_forces;
    Cuda_Buffer<scalar>* cb_mass;
    Cuda_Buffer<scalar>* cb_inv_mass;

    // Mesh
    Cuda_Buffer<int>* cb_topology;
    Cuda_Buffer<int>* cb_offsets;
    Cuda_Buffer<Matrix3x3>* cb_JX_inv;
    Cuda_Buffer<Vector3>* cb_dN;
    Cuda_Buffer<scalar>* cb_weights;
    Cuda_Buffer<scalar>* cb_V;

    GPU_PBD_FEM(Element element, const std::vector<Vector3>& geometry, const std::vector<int>& topology, const std::vector<int>& offsets, float density);

    ~GPU_PBD_FEM() = default;

    void step(scalar dt) const;

    void get_position(std::vector<Vector3>& p) const {cb_position->get_data(p);}
    void get_velocity(std::vector<Vector3>& v) const {cb_velocity->get_data(v);}
    void get_forces(std::vector<Vector3>& f) const {cb_forces->get_data(f);}
    void get_prev_position(std::vector<Vector3>& p) const {cb_prev_position->get_data(p);}
    void get_mass(std::vector<scalar>& m) const {cb_mass->get_data(m);}
    void get_inv_mass(std::vector<scalar>& w) const {cb_inv_mass->get_data(w);};
};

__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3* pos, Vector3* dN);
__global__ void kernel_constraint_solve(int n, int nb_quadrature, int nb_vert_elem, int offset, Vector3* p, int* topology, Vector3* dN, scalar* V, Matrix3x3* JX_inv);
__global__ void kernel_velocity_update(int n, float dt, Vector3* p, Vector3* prev_p, Vector3* v);
__global__ void kernel_step_solver(int n, float dt, Vector3 g, Vector3* p, Vector3* v, Vector3* f, float* w);
__global__ void kernel_constraint_plane(int n, float dt, Vector3 origin, Vector3 normal, Vector3* p, Vector3* v, Vector3* f);