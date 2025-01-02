#pragma once
#include <Dynamic/VBD/VBD_FEM.h>
#include "VBD_FEM_Dynamic.h"

struct MG_VBD_FEM_Dynamic final : VBD_FEM_Dynamic
{
    explicit MG_VBD_FEM_Dynamic(
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, const Material material,
        const int iteration = 30, const int sub_iteration = 1, const scalar damping = 0.f, const scalar rho = 0.f,
        const scalar linear = 0.5f)
        : VBD_FEM_Dynamic(density, m_distrib, young, poisson, material, iteration, sub_iteration, damping, rho),
          _linear(linear)
    {
    }

    ParticleSystem* build_particle_system() override;

    void build_dynamic() override;

protected:
    scalar _linear;
};
