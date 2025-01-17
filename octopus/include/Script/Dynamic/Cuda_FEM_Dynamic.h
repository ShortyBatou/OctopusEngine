#pragma once
#include <GPU/Explicit/GPU_Explicit.h>

#include "Core/Base.h"
#include "Cuda_ParticleSystem_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"
#include "Manager/TimeManager.h"

struct Cuda_FEM_Dynamic : public Cuda_ParticleSystem_Dynamics, public FEM_Dynamic_Getters
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
            auto* fem = new GPU_Explicit_FEM(e, _mesh->geometry(), topo, _young, _poisson, _material, _damping);
            _gpu_fems[e] = fem;
            _gpu_ps->add_dynamics(fem);
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


    [[nodiscard]] std::map<Element, std::vector<scalar>> get_stress() override
    {
        std::map<Element, std::vector<scalar>> stresses;
        for(auto&[e, topo] : _mesh->topologies())
        {
            if(topo.empty()) continue;
            stresses[e] = _gpu_fems[e]->get_stress(_gpu_ps);
        }
        return stresses;
    }

    [[nodiscard]] std::map<Element, std::vector<scalar>> get_volume() override
    {
        std::map<Element, std::vector<scalar>> volumes;
        for(auto&[e, topo] : _mesh->topologies())
        {
            if(topo.empty()) continue;
            volumes[e] = _gpu_fems[e]->get_volume(_gpu_ps);
        }
        return volumes;
    }

    [[nodiscard]] std::map<Element, std::vector<scalar>> get_volume_diff() override
    {
        std::map<Element, std::vector<scalar>> volumes;
        for(auto&[e, topo] : _mesh->topologies())
        {
            if(topo.empty()) continue;
            volumes[e] = _gpu_fems[e]->get_volume_diff(_gpu_ps);
        }
        return volumes;
    }

    [[nodiscard]] std::vector<scalar> get_stress_vertices() override
    {
        std::vector<scalar> stresses(_gpu_ps->nb_particles(),0);
        for(auto&[e, topo] : _mesh->topologies())
        {
            if(topo.empty()) continue;
            std::vector<scalar> stress = _gpu_fems[e]->get_stress(_gpu_ps);
            const int nb_vert_elem = elem_nb_vertices(e);
            const int nb_elem = static_cast<int>(topo.size()) / nb_vert_elem;
            for(int i = 0; i < nb_elem; i++)
            {
                const int eid = i * nb_vert_elem;
                for(int j = 0; j < nb_vert_elem; j++)
                    stresses[topo[eid+j]] += stress[i];
            }
        }
        return stresses;
    }

    [[nodiscard]] std::vector<Vector3> get_residual_vertices() override {
        std::vector<Vector3> residuals(_gpu_ps->nb_particles(), Vector3(0));
        scalar sub_dt = Time::Fixed_DeltaTime() / static_cast<scalar>(_sub_iterations);
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            std::vector<Vector3> r = _gpu_fems[e]->get_forces(_gpu_ps, sub_dt);
            const int nb_vert_elem = elem_nb_vertices(e);
            const int nb_elem = static_cast<int>(topo.size()) / nb_vert_elem;
            for(int i = 0; i < nb_elem; i++)
            {
                const int eid = i * nb_vert_elem;
                for(int j = 0; j < nb_vert_elem; j++)
                    residuals[topo[eid+j]] += r[topo[eid+j]];
            }
        }
        return residuals;
    }

    std::map<Element, GPU_FEM*> _gpu_fems;

    Mass_Distribution _m_distrib;
    scalar _young, _poisson;
    scalar _damping;
    Material _material;

};
