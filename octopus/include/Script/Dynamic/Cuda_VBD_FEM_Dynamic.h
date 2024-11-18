#pragma once
#include "Core/Base.h"
#include "GPU/GPU_VBD.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector

struct Cuda_VBD_FEM_Dynamic final : Component
{
    explicit Cuda_VBD_FEM_Dynamic(
        const scalar density, const Mass_Distribution m_distrib,
        const scalar young, const scalar poisson,
        const int iteration = 30, const int sub_iteration=1, const scalar damping = 0.f)
        : _density(density), _m_distrib(m_distrib), _damping(damping), _young(young), _poisson(poisson),
          _iteration(iteration), _sub_iteration(sub_iteration),
          _mesh(nullptr), vbd(nullptr)
    { }

    void init() override;

    void update() override;

    ~Cuda_VBD_FEM_Dynamic() override = default;


private:
    std::map<Element, std::vector<Color>> _display_colors;
    scalar _density;
    Mass_Distribution _m_distrib;
    scalar _damping;
    scalar _young, _poisson;
    int _iteration;
    int _sub_iteration;
    GPU_VBD* vbd;
    Mesh* _mesh;
};
