#pragma once
#include <GPU/GPU_Explicit.h>

#include "Core/Base.h"
#include "Cuda_ParticleSystem_Dynamic.h"


struct Cuda_FEM_Dynamic : Cuda_ParticleSystem_Dynamics
{
    explicit Cuda_FEM_Dynamic(int sub_it,
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, const Material material, scalar damping) :
        Cuda_ParticleSystem_Dynamics(sub_it, density), _m_distrib(m_distrib),
              _young(young), _poisson(poisson), _material(material), _damping(damping)
    {}

    GPU_ParticleSystem* create_particle_system() override
    {
        return new GPU_ParticleSystem(_mesh->geometry(), get_fem_masses(), new GPU_SemiExplicit(), _sub_iterations);
    }

    void build_dynamics() override
    {
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            // create CUDA FEM Explicit
            _gpu_ps->add_dynamics(new GPU_Explicit_FEM(e, _mesh->geometry(), topo, _young, _poisson, _material, _damping));
        }
    }

    [[nodiscard]] std::vector<scalar> get_fem_masses() const
    {
        std::vector masses(_mesh->nb_vertices(),0.f);
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            const std::vector<scalar> e_masses = compute_fem_mass(e, _mesh->geometry(),topo, _density, _m_distrib); // depends on density
            for(size_t i = 0; i < e_masses.size(); i++)
                masses[i] += e_masses[i];
        }
        return masses;
    }

    Mass_Distribution _m_distrib;
    scalar _young, _poisson;
    scalar _damping;
    Material _material;
};
