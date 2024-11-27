#pragma once
#include "Core/Base.h"
#include "GPU/PBD/GPU_PBD.h"
#include "GPU/PBD/GPU_PBD_FEM.h"
#include "Cuda_FEM_Dynamic.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector

struct Cuda_XPBD_FEM_Dynamic final : Cuda_FEM_Dynamic
{
    explicit Cuda_XPBD_FEM_Dynamic(
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, const Material material,
        const int sub_it = 30, const scalar damping = 0.f, const bool coupled = false)
        : Cuda_FEM_Dynamic(sub_it, density, m_distrib, young, poisson, material, damping),
            _coupled_fem(coupled)
    { }


    void update() override;

    GPU_ParticleSystem* create_particle_system() override;
    void build_dynamics() override;

private:
    std::map<Element, std::vector<Color>> _display_colors;
    std::map<Element, GPU_PBD_FEM*> _gpu_fems;
    bool _coupled_fem;
};
