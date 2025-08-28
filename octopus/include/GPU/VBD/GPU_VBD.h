#pragma once
#include <set>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Mesh/Mesh.h>

#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/VBD/GPU_VBD_FEM.h"

struct GPU_VBD : GPU_ParticleSystem
{
    GPU_VBD(const std::vector<Vector3>& positions, const std::vector<scalar>& masses, const int it, const int sub_it,
            const scalar rho)
        : GPU_ParticleSystem(positions, masses, nullptr, sub_it), iteration(it), _rho(rho)
    {
        y = new Cuda_Buffer(positions);
        prev_it_p = new Cuda_Buffer(positions);
        prev_it2_p = new Cuda_Buffer(positions);
        prev_prev_p = new Cuda_Buffer(positions);
    }

    void step(scalar dt) override;

    void add_dynamics(GPU_Dynamic* dynamic) override
    {
        GPU_VBD_FEM* _fem = dynamic_cast<GPU_VBD_FEM*>(dynamic);
        if (_fem != nullptr) _fem->y = y; // ugly as fuck
        GPU_ParticleSystem::add_dynamics(dynamic);
    }

    int iteration;

    scalar _rho;
    Cuda_Buffer<Vector3>* prev_it_p;
    Cuda_Buffer<Vector3>* prev_it2_p;

    Cuda_Buffer<Vector3>* y;
    Cuda_Buffer<Vector3>* prev_prev_p;

    ~GPU_VBD() override
    {
        delete prev_it2_p;
        delete prev_it_p;
        delete y;
    }
};

__global__ void kernel_integration(scalar dt, Vector3 g, GPU_ParticleSystem_Parameters ps, Vector3* y,
                                   Vector3* prev_it_p, Vector3* prev_prev_p);
__global__ void kernel_velocity_update(scalar dt, GPU_ParticleSystem_Parameters ps);
__global__ void kernel_chebychev_acceleration(int it, scalar omega, GPU_ParticleSystem_Parameters ps,
                                              Vector3* prev_it_p, Vector3* prev_it2_p);
