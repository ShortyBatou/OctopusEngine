#pragma once
#include <set>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Mesh/Mesh.h>

#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/VBD/GPU_VBD.h"
#include "GPU/VBD/GPU_MG_VBD_FEM.h"

struct GPU_MG_VBD : GPU_ParticleSystem
{
    GPU_MG_VBD(const std::vector<Vector3>& positions, const std::vector<scalar>& masses, const int it, const int sub_it)
        : GPU_ParticleSystem(positions, masses, nullptr, sub_it), iteration(it)
    {
        y = new Cuda_Buffer(positions);
    }

    void step(scalar dt) override;

    void add_dynamics(GPU_Dynamic* dynamic) override
    {
        auto* _fem = dynamic_cast<GPU_MG_VBD_FEM*>(dynamic);
        if (_fem != nullptr)
        {
            fems.push_back(_fem);
            _fem->y = y; // ugly as fuck
        }
        GPU_ParticleSystem::add_dynamics(dynamic);
    }

    int iteration;
    Cuda_Buffer<Vector3>* y;
    std::vector<GPU_MG_VBD_FEM*> fems;
    ~GPU_MG_VBD() override
    {
        delete y;
    }
};
