#pragma once
#include "Core/Base.h"
#include <Dynamic/FEM/FEM_Shape.h>
#include "GPU/Cuda_Buffer.h"
#include "GPU/VBD/GPU_VBD.h"
#include "GPU/VBD/GPU_Mixed_VBD_FEM.h"

struct GPU_Mixed_VBD final : GPU_VBD {
    GPU_Mixed_VBD(const std::vector<Vector3>& positions, const std::vector<scalar>& masses, const int it, const int sub_it, const int exp_it)
    : GPU_VBD(positions, masses, it, sub_it, 0)
    {
        explicit_it = exp_it;
    }

    void step(scalar dt) override;

    void add_dynamics(GPU_Dynamic* dynamic) override
    {
        if (auto* _fem = dynamic_cast<GPU_Mixed_VBD_FEM*>(dynamic); _fem != nullptr) _fems.push_back(_fem);
        GPU_VBD::add_dynamics(dynamic);
    }

    std::vector<GPU_Mixed_VBD_FEM*> _fems;
    int explicit_it;
    ~GPU_Mixed_VBD() override = default;
};
