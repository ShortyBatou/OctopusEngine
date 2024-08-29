#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include <Dynamic/FEM/FEM_Shape.h>
#include "GPU/Cuda_Buffer.h"
#include <Manager/Dynamic.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct GPU_ParticleSystem {
    int n;
    Cuda_Buffer<Vector3>* cb_position;
    Cuda_Buffer<Vector3>* cb_prev_position;
    Cuda_Buffer<Vector3>* cb_init_position;
    Cuda_Buffer<Vector3>* cb_velocity;
    Cuda_Buffer<Vector3>* cb_forces;
    Cuda_Buffer<scalar>* cb_mass;
    Cuda_Buffer<scalar>* cb_inv_mass;

    GPU_ParticleSystem(const std::vector<Vector3>& positions, const std::vector<scalar>& masses);
    ~GPU_ParticleSystem() = default;

    [[nodiscard]] int nb() const { return n; }
    void get_position(std::vector<Vector3>& p) const {cb_position->get_data(p);}
    void get_prev_position(std::vector<Vector3>& p) const {cb_prev_position->get_data(p);}
    void get_velocity(std::vector<Vector3>& v) const {cb_velocity->get_data(v);}
    void get_forces(std::vector<Vector3>& f) const {cb_forces->get_data(f);}
    void get_mass(std::vector<scalar>& m) const {cb_mass->get_data(m);}
    void get_inv_mass(std::vector<scalar>& w) const {cb_inv_mass->get_data(w);}
};