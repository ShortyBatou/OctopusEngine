#pragma once

#include "Script/Dynamic/Cuda_FEM_Dynamic.h"

struct Cuda_FEM_Explicit_Dynamic : Cuda_FEM_Dynamic
{
    explicit Cuda_FEM_Explicit_Dynamic(int sub_it,
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, const Material material, scalar damping, IntegratorType integrator) :
        Cuda_FEM_Dynamic(sub_it,density,m_distrib,young,poisson,material,damping), integrator_type(integrator)
    {}

    GPU_Integrator* create_integrator() override
    {
        switch (integrator_type) {
            case RK2: return new GPU_Explicit_RK2(_mesh->geometry().size());
            case RK4: return new GPU_Explicit_RK4(_mesh->geometry().size());
            default: return new GPU_SemiExplicit();
        }

    }



    IntegratorType integrator_type;

};
