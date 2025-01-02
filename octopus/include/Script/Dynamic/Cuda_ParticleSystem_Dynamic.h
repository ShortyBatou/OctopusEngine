#pragma once
#include "Core/Base.h"
#include "GPU/PBD/GPU_PBD.h"
#include "GPU/PBD/GPU_PBD_FEM.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector
#include <Dynamic/Base/ParticleSystem.h>
#include <Manager/Debug.h>
#include "ParticleSystemDynamic.h"

struct Cuda_ParticleSystem_Dynamics : Component, ParticleSystemDynamics_Getters
{
    Cuda_ParticleSystem_Dynamics(const int sub_iterations, const scalar density)
        : _density(density), _sub_iterations(sub_iterations), _gpu_ps(nullptr), _mesh(nullptr) { }

    void init() override;
    void update() override;

    [[nodiscard]] GPU_ParticleSystem* get_particle_system() const { return _gpu_ps;}

    ~Cuda_ParticleSystem_Dynamics() override { delete _gpu_ps; }

    [[nodiscard]] std::vector<Vector3> get_positions() override;
    [[nodiscard]] std::vector<Vector3> get_init_positions() override;
    [[nodiscard]] std::vector<Vector3> get_displacement() override;
    [[nodiscard]] std::vector<Vector3> get_velocity() override;
    [[nodiscard]] std::vector<scalar> get_masses() override;
    [[nodiscard]] std::vector<scalar> get_massses_inv() override;
    [[nodiscard]] std::vector<scalar> get_displacement_norm() override;
    [[nodiscard]] std::vector<scalar> get_velocity_norm() override;

protected:
    virtual GPU_ParticleSystem* create_particle_system()
    {
        const std::vector<scalar> masses(_mesh->nb_vertices(), _density);
        return new GPU_ParticleSystem(_mesh->geometry(), masses, new GPU_SemiExplicit(), _sub_iterations);
    }
    virtual void build_dynamics() { }

    scalar _density;
    int _sub_iterations;
    GPU_ParticleSystem* _gpu_ps;
    Mesh* _mesh;
};
