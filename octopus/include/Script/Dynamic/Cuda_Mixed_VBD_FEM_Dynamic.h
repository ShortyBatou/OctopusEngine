#pragma once
#include "Core/Base.h"
#include "GPU/VBD/GPU_VBD.h"
#include "Cuda_FEM_Dynamic.h"
#include "Script/Dynamic/Cuda_VBD_FEM_Dynamic.h"
#include<vector> // for vector


struct Cuda_Mixed_VBD_FEM_Dynamic final : Cuda_VBD_FEM_Dynamic
{
    explicit Cuda_Mixed_VBD_FEM_Dynamic(
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, Material material,
        const int iteration = 30, const int sub_iteration=1, const int exp_it = 1, const scalar damping = 0.f)
        : Cuda_VBD_FEM_Dynamic(density, m_distrib, young, poisson, material, iteration, sub_iteration, damping, 0), _exp_it(exp_it)
    { }

    GPU_ParticleSystem* create_particle_system() override;
    void build_dynamics() override;

    ~Cuda_Mixed_VBD_FEM_Dynamic() override = default;
private:
    int _exp_it;
};
