#pragma once
#include "Core/Base.h"
#include "GPU/GPU_VBD.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector

struct Cuda_VBD_FEM_Dynamic final : Component
{
    explicit Cuda_VBD_FEM_Dynamic(
        const scalar density,
        const scalar young, const scalar poisson,
        const int iteration = 30, const scalar damping = 0.f)
        : _density(density), _damping(damping), _young(young), _poisson(poisson), _iteration(iteration),
        _mesh(nullptr)
    {
    }

    void init() override;

    void update() override;

    ~Cuda_VBD_FEM_Dynamic() override = default;

    void add_dynamic(GPU_Dynamic* dynamic) {

    }

private:
    std::map<Element, std::vector<Color>> _display_colors;
    scalar _density;
    scalar _damping;
    scalar _young, _poisson;
    int _iteration;

    GPU_VBD* vbd;
    Mesh* _mesh;
};
