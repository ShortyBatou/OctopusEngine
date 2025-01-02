#pragma once
#include "Core/Base.h"
#include "GPU/VBD/GPU_VBD.h"
#include "Cuda_FEM_Dynamic.h"
#include<vector> // for vector

#include "Cuda_VBD_FEM_Dynamic.h"


struct Cuda_MG_VBD_FEM_Dynamic : Cuda_VBD_FEM_Dynamic
{
    explicit Cuda_MG_VBD_FEM_Dynamic(
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, Material material,
        const int iteration = 30, const int sub_iteration=1, const scalar damping = 0.f, const scalar rho = 0.f, const scalar linear = 0.f)
        : Cuda_VBD_FEM_Dynamic(density, m_distrib, young, poisson, material, iteration, sub_iteration, damping, rho), _linear(linear)
    { }


    GPU_ParticleSystem* create_particle_system() override;
    void build_dynamics() override;

    ~Cuda_MG_VBD_FEM_Dynamic() override = default;

    scalar _linear;
};
