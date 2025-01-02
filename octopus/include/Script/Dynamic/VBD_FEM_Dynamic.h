#pragma once
#include "Core/Base.h"
#include "Dynamic/VBD/VertexBlockDescent.h"
#include "Mesh/Mesh.h"
#include<vector> // for vector
#include <Dynamic/VBD/VBD_FEM.h>

#include "FEM_Dynamic.h"
#include "ParticleSystemDynamic.h"


struct VBD_FEM_Dynamic : FEM_Dynamic
{
    explicit VBD_FEM_Dynamic(
        const scalar density, Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, const Material material,
        const int iteration = 30, const int sub_iteration = 1, const scalar damping = 0.f, const scalar rho = 0.f)
        : FEM_Dynamic(density, m_distrib, young, poisson, material, sub_iteration),
          fem(nullptr), _iteration(iteration), _rho(rho), _damping(damping), vbd(nullptr)
    {
    }

    void update() override;

    ~VBD_FEM_Dynamic() override = default;
    ParticleSystem* build_particle_system() override;

    void build_dynamic() override;

    [[nodiscard]] std::map<Element, std::vector<scalar>> get_stress() override;
    [[nodiscard]] std::map<Element, std::vector<scalar>> get_volume() override;
    [[nodiscard]] std::map<Element, std::vector<scalar>> get_volume_diff() override;

protected:
    VBD_FEM* fem;
    std::map<Element, std::vector<Color>> _display_colors;
    int _iteration;
    scalar _rho, _damping;
    VertexBlockDescent* vbd;
};
