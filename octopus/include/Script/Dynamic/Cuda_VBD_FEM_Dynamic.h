#pragma once
#include "Core/Base.h"
#include "GPU/VBD/GPU_VBD.h"
#include "Cuda_FEM_Dynamic.h"
#include<vector> // for vector


struct Cuda_VBD_FEM_Dynamic : Cuda_FEM_Dynamic
{
    explicit Cuda_VBD_FEM_Dynamic(
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson, Material material,
        const int iteration = 30, const int sub_iteration=1, const scalar damping = 0.f, const scalar rho = 0.f, const VBD_Version version = VBD_Version::Better_Coloration, const int target_color = -1)
        : Cuda_FEM_Dynamic(sub_iteration, density, m_distrib, young, poisson, material, damping), _rho(rho), _iteration(iteration), _version(version), _target_color(target_color)
    { }

    GPU_Integrator* Cuda_VBD_FEM_Dynamic::create_integrator() override;
    void build_dynamics() override;
    void update() override;

    ~Cuda_VBD_FEM_Dynamic() override = default;
protected:
    std::map<Element, std::vector<Color>> _display_colors;
    float _rho;
    VBD_Version _version;
    int _target_color;
    int _iteration;
};
