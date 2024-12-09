#pragma once
#include "Core/Base.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU_Dynamic.h"
#include "GPU_Integrator.h"

struct GPU_ParticleSystem
{
    GPU_ParticleSystem(const std::vector<Vector3>& positions, const std::vector<scalar>& masses,
                       GPU_Integrator* integrator, int _sub_iteration);

    virtual void step(scalar dt);

    [[nodiscard]] int nb_particles() const { return _nb_particles; }
    void get_position(std::vector<Vector3>& p) const { _cb_position->get_data(p); }
    void get_prev_position(std::vector<Vector3>& p) const { _cb_prev_position->get_data(p); }
    void get_velocity(std::vector<Vector3>& v) const { _cb_velocity->get_data(v); }
    void get_forces(std::vector<Vector3>& f) const { _cb_forces->get_data(f); }
    void get_mass(std::vector<scalar>& m) const { _cb_mass->get_data(m); }
    void get_inv_mass(std::vector<scalar>& w) const { _cb_inv_mass->get_data(w); }

    virtual void add_dynamics(GPU_Dynamic* dynamic) { _dynamics.push_back(dynamic); }
    virtual void add_constraint(GPU_Dynamic* constraint) { _constraints.push_back(constraint); }
    [[nodiscard]] Vector3* buffer_position() const { return _cb_position->buffer; }
    [[nodiscard]] Vector3* buffer_prev_position() const { return _cb_prev_position->buffer; }
    [[nodiscard]] Vector3* buffer_init_position() const { return _cb_init_position->buffer; }
    [[nodiscard]] Vector3* buffer_velocity() const { return _cb_velocity->buffer; }
    [[nodiscard]] Vector3* buffer_forces() const { return _cb_forces->buffer; }
    [[nodiscard]] scalar* buffer_mass() const { return _cb_mass->buffer; }
    [[nodiscard]] scalar* buffer_inv_mass() const { return _cb_inv_mass->buffer; }
    [[nodiscard]] int* buffer_mask() const { return _cb_mask->buffer; }

    virtual ~GPU_ParticleSystem()
    {
        delete _cb_position;
        delete _cb_prev_position;
        delete _cb_init_position;
        delete _cb_velocity;
        delete _cb_forces;
        delete _cb_mass;
        delete _cb_inv_mass;
        delete _cb_mask;

        delete _integrator;
        for(GPU_Dynamic* dynamic: _dynamics) delete dynamic;
        for(GPU_Dynamic* dynamic: _constraints) delete dynamic;
    }


public:
    int _sub_iteration;
protected:
    GPU_Integrator* _integrator;
    std::vector<GPU_Dynamic*> _dynamics;
    std::vector<GPU_Dynamic*> _constraints;

    int _nb_particles;
    Cuda_Buffer<Vector3>* _cb_position;
    Cuda_Buffer<Vector3>* _cb_prev_position;
    Cuda_Buffer<Vector3>* _cb_init_position;
    Cuda_Buffer<Vector3>* _cb_velocity;
    Cuda_Buffer<Vector3>* _cb_forces;
    Cuda_Buffer<scalar>* _cb_mass;
    Cuda_Buffer<scalar>* _cb_inv_mass;
    Cuda_Buffer<int>* _cb_mask;
};
