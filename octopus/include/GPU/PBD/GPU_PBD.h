#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "cuda_runtime.h"
#include "GPU/GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_PBD final : GPU_ParticleSystem
{
    GPU_PBD(const std::vector<Vector3>& positions, const std::vector<scalar>& masses, const int sub_iteration,
            const scalar global_damping = 0.f)
        : GPU_ParticleSystem(positions, masses, new GPU_SemiExplicit(), sub_iteration),
            _global_damping(global_damping)
    {
    }

    void step(scalar dt) override;

    scalar _global_damping;
    ~GPU_PBD() override = default;
};

__global__ void kernel_velocity_update(int n, float dt, scalar global_damping, GPU_ParticleSystem_Parameters ps);
