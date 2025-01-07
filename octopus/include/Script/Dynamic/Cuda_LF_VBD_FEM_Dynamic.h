#pragma once
#include "Core/Base.h"
#include "GPU/VBD/GPU_VBD.h"
#include "Script/Dynamic/Cuda_VBD_FEM_Dynamic.h"
#include<vector> // for vector


struct Cuda_LF_VBD_FEM_Dynamic final : Cuda_VBD_FEM_Dynamic
{
    explicit Cuda_LF_VBD_FEM_Dynamic(
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, Material material,
        const int iteration = 30, const int sub_iteration=1, const scalar damping = 0.f, const scalar rho = 0.f)
        : Cuda_VBD_FEM_Dynamic(density, m_distrib, young, poisson, material, iteration, sub_iteration, damping, rho)
    { }

    void build_dynamics() override;

    ~Cuda_LF_VBD_FEM_Dynamic() override = default;
};
