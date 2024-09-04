#pragma once
#include "Core/Base.h"
#include "GPU/GPU_PBD.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector

struct Cuda_XPBD_FEM_Dynamic final : Component
{
    explicit Cuda_XPBD_FEM_Dynamic(
        const scalar density,
        const scalar young, const scalar poisson,
        const int iteration = 30, const scalar damping = 0.f)
        : _density(density), _damping(damping), _young(young), _poisson(poisson), _iteration(iteration),
        _mesh(nullptr), _gpu_pbd(nullptr)
    {
    }

    void init() override;

    void update() override;

    ~Cuda_XPBD_FEM_Dynamic() override
    {
        delete _gpu_pbd;
    }

    void add_dynamic(GPU_Dynamic* dynamic)
    {
        _gpu_pbd->dynamic.push_back(dynamic);
    }

private:
    std::map<Element, std::vector<Color>> _display_colors;
    std::map<Element, GPU_PBD_FEM*> _gpu_fems;
    scalar _density;
    scalar _damping;
    scalar _young, _poisson;
    int _iteration;

    Mesh* _mesh;
    GPU_PBD* _gpu_pbd;
};
