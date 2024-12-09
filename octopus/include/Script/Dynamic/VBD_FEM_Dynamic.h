#pragma once
#include "Core/Base.h"
#include "Dynamic/VBD/VertexBlockDescent.h"
#include "Mesh/Mesh.h"
#include<vector> // for vector

#include "ParticleSystemDynamic.h"

struct VBD_FEM_Dynamic final : ParticleSystemDynamic
{
    explicit VBD_FEM_Dynamic(
        const scalar density, Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, const Material material,
        const int iteration = 30, const int sub_iteration=1, const scalar damping = 0.f, const scalar rho = 0.f)
        : ParticleSystemDynamic(density),
            _density(density), _m_distrib(m_distrib), _damping(damping), _rho(rho),
            _young(young), _poisson(poisson), _material(material),
            _iteration(iteration), _sub_iteration(sub_iteration),
            vbd(nullptr)
    {

    }

    void update() override;

    ~VBD_FEM_Dynamic() override = default;
    ParticleSystem *build_particle_system() override;

    void build_dynamic() override;

private:
    std::map<Element, std::vector<Color>> _display_colors;
    scalar _density;
    Mass_Distribution _m_distrib;
    scalar _damping, _rho;
    scalar _young, _poisson;
    Material _material;
    int _iteration;
    int _sub_iteration;
    VertexBlockDescent* vbd;
};
