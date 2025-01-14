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
        w_max = new Cuda_Buffer<scalar>(std::vector<scalar>(positions.size()));
        l = new Cuda_Buffer<Vector3>(std::vector<Vector3>(positions.size() * 4));
        k = new Cuda_Buffer<Vector3>(std::vector<Vector3>(positions.size() * 4));
        last_v = new Cuda_Buffer<Vector3>(std::vector<Vector3>(positions.size()));
        rk4_last_p = new Cuda_Buffer<Vector3>(std::vector<Vector3>(positions.size()));

        const scalar f = 0.01;
        const scalar m_max = *std::max_element(masses.begin(), masses.end());
        const scalar thresh = m_max * f;
        std::vector<int> masks(masses.size());
        for(int i = 0; i < masses.size(); ++i) {
            if(masses[i] >= thresh) masks[i] = 1;
            else masks[i] = 3;
        }
        this->_data->_cb_mask->load_data(masks);

    }

    void step(scalar dt) override;

    void add_dynamics(GPU_Dynamic* dynamic) override
    {
        if (auto* _fem = dynamic_cast<GPU_Mixed_VBD_FEM*>(dynamic); _fem != nullptr) _fems.push_back(_fem);
        GPU_VBD::add_dynamics(dynamic);
    }

    std::vector<GPU_Mixed_VBD_FEM*> _fems;
    Cuda_Buffer<scalar>* w_max;
    Cuda_Buffer<Vector3>* l;
    Cuda_Buffer<Vector3>* k;
    Cuda_Buffer<Vector3>* last_v;
    Cuda_Buffer<Vector3>* rk4_last_p;

    int explicit_it;
    ~GPU_Mixed_VBD() override
    {
        delete l;
        delete k;
        delete last_v;
        delete w_max;
    }
};
