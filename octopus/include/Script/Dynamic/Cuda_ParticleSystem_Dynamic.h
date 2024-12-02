#pragma once
#include "Core/Base.h"
#include "GPU/PBD/GPU_PBD.h"
#include "GPU/PBD/GPU_PBD_FEM.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector
#include <Dynamic/Base/ParticleSystem.h>
#include <Manager/Debug.h>

struct Cuda_ParticleSystem_Dynamics : Component
{
    Cuda_ParticleSystem_Dynamics(const int sub_iterations, const scalar density)
        : _density(density), _sub_iterations(sub_iterations), _gpu_ps(nullptr), _mesh(nullptr) { }

    void init() override
    {
        _mesh = this->entity()->get_component<Mesh>();
        _gpu_ps = create_particle_system();
        build_dynamics();
    }

    void update() override
    {
        Time::Tic();
        const scalar sub_dt = Time::Fixed_DeltaTime() / static_cast<scalar>(_sub_iterations);
        for(int i = 0; i < _sub_iterations; i++)
            _gpu_ps->step(sub_dt);
        scalar t = Time::Tac() * 1000.f;
        _gpu_ps->get_position(_mesh->geometry());

        DebugUI::Begin("Entity " + std::to_string(entity()->id()));
        DebugUI::Plot("Time (" + std::to_string(entity()->id()) + ")", t);
        DebugUI::Range("Range (" + std::to_string(entity()->id()) + ")", t);
        DebugUI::Value("Value (" + std::to_string(entity()->id()) + ")", t);
        DebugUI::End();
    }

    [[nodiscard]] GPU_ParticleSystem* get_particle_system() const
    {
        return _gpu_ps;
    }

    ~Cuda_ParticleSystem_Dynamics() override { delete _gpu_ps; }

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
