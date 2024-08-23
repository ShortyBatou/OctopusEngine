#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "Tools/Random.h"
#include "Tools/Interpolation.h"
struct GPU_PB_FEM {
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

    GPU_PB_FEM(Element element, const std::vector<Vector3>& geometry, const std::vector<int>& topology, const std::vector<int>& offsets, float density);
    ~GPU_PB_FEM();

    void step(scalar dt);

    void get_position(std::vector<Vector3>& p) const {cb_position->get_data(p);}
    void get_velocity(std::vector<Vector3>& v) const {cb_velocity->get_data(v);}
    void get_forces(std::vector<Vector3>& f) const {cb_forces->get_data(f);}
    void get_prev_position(std::vector<Vector3>& p) const {cb_prev_position->get_data(p);}
    void get_mass(std::vector<scalar>& m) const {cb_mass->get_data(m);}
    void get_inv_mass(std::vector<scalar>& w) const {cb_inv_mass->get_data(w);};


};