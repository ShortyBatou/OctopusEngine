#pragma once
#include "Core/Base.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU_Dynamic.h"
#include "GPU_Integrator.h"

// struct used as parameter for cuda kernels
struct GPU_ParticleSystem_Parameters
{
    GPU_ParticleSystem_Parameters() = default;
    ~GPU_ParticleSystem_Parameters() = default;

    int nb_particles;
    Vector3* p;
    Vector3* last_p;
    Vector3* init_p;
    Vector3* v;
    Vector3* f;
    scalar* m;
    scalar* w;
    int* mask;
};

// struct used to store and handle the memory
struct GPU_ParticleSystem_Data final
{
    GPU_ParticleSystem_Data() : _nb_particles(0),
        _cb_position(nullptr), _cb_prev_position(nullptr), _cb_init_position(nullptr),
        _cb_velocity(nullptr), _cb_forces(nullptr), _cb_mass(nullptr), _cb_inv_mass(nullptr), _cb_mask(nullptr) {}


    int _nb_particles;
    Cuda_Buffer<Vector3>* _cb_position;
    Cuda_Buffer<Vector3>* _cb_prev_position;
    Cuda_Buffer<Vector3>* _cb_init_position;
    Cuda_Buffer<Vector3>* _cb_velocity;
    Cuda_Buffer<Vector3>* _cb_forces;
    Cuda_Buffer<scalar>* _cb_mass;
    Cuda_Buffer<scalar>* _cb_inv_mass;
    Cuda_Buffer<int>* _cb_mask;

    virtual ~GPU_ParticleSystem_Data()
    {
        delete _cb_position;
        delete _cb_prev_position;
        delete _cb_init_position;
        delete _cb_velocity;
        delete _cb_forces;
        delete _cb_mass;
        delete _cb_inv_mass;
        delete _cb_mask;
    }
};



struct GPU_ParticleSystem
{
    GPU_ParticleSystem(const std::vector<Vector3>& positions, const std::vector<scalar>& masses,
                       GPU_Integrator* integrator, int _sub_iteration);

    virtual void step(scalar dt);

    [[nodiscard]] int nb_particles() const { return _data->_nb_particles; }
    void get_position(std::vector<Vector3>& p) const { _data->_cb_position->get_data(p); }
    void get_prev_position(std::vector<Vector3>& p) const { _data->_cb_prev_position->get_data(p); }
    void get_velocity(std::vector<Vector3>& v) const { _data->_cb_velocity->get_data(v); }
    void get_forces(std::vector<Vector3>& f) const { _data->_cb_forces->get_data(f); }
    void get_mass(std::vector<scalar>& m) const { _data->_cb_mass->get_data(m); }
    void get_inv_mass(std::vector<scalar>& w) const { _data->_cb_inv_mass->get_data(w); }
    void get_init_position(std::vector<Vector3>& p) const { _data->_cb_init_position->get_data(p); }

    [[nodiscard]] GPU_ParticleSystem_Parameters get_parameters() const;

    virtual void add_dynamics(GPU_Dynamic* dynamic) { _dynamics.push_back(dynamic); }
    virtual void add_constraint(GPU_Dynamic* constraint) { _constraints.push_back(constraint); }

    virtual ~GPU_ParticleSystem()
    {
        delete _integrator;
        for(const GPU_Dynamic* dynamic: _dynamics) delete dynamic;
        for(const GPU_Dynamic* dynamic: _constraints) delete dynamic;
    }

public:
    int _sub_iteration;
protected:
    GPU_Integrator* _integrator;
    std::vector<GPU_Dynamic*> _dynamics;
    std::vector<GPU_Dynamic*> _constraints;
    GPU_ParticleSystem_Data* _data;
};
