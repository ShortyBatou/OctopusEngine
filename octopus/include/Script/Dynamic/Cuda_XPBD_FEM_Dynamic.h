#pragma once
#include "Core/Base.h"
#include "GPU/GPU_PBD.h"
#include "GPU/GPU_PBD_FEM.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector

struct Cuda_XPBD_FEM_Dynamic final : Component
{
    explicit Cuda_XPBD_FEM_Dynamic(
        const scalar density,
        const scalar young, const scalar poisson, const Material material,
        const int iteration = 30, const scalar damping = 0.f, const bool coupled = false)
        : _density(density), _damping(damping), _young(young), _poisson(poisson), _material(material), _iteration(iteration),
        _mesh(nullptr), _gpu_pbd(nullptr), _coupled_fem(coupled)
    {
    }

    void init() override;

    void update() override;

    ~Cuda_XPBD_FEM_Dynamic() override
    {
        delete _gpu_pbd;
    }

    void add_dynamic(GPU_Dynamic* dynamic) const {
        _gpu_pbd->dynamic.push_back(dynamic);
    }

    void set_sub_iteration(const int it) {
        _iteration = it;
        _gpu_pbd->iteration = it;
    }

    [[nodiscard]] int get_sub_iteration() const {
        return _iteration;
    }

    void set_damping(const scalar damping) {
        _damping = damping;
        _gpu_pbd->global_damping = damping;
    }

private:
    std::map<Element, std::vector<Color>> _display_colors;
    std::map<Element, GPU_PBD_FEM*> _gpu_fems;
    scalar _density;
    scalar _damping;
    scalar _young, _poisson;
    Material _material;
    int _iteration;
    bool _coupled_fem;
    Mesh* _mesh;
    GPU_PBD* _gpu_pbd;
};
