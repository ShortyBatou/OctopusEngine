#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "cuda_runtime.h"
#include "GPU/GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_PBD final : GPU_Integrator
{
    GPU_PBD(const scalar global_damping = 0.f)
        : _global_damping(global_damping)
    {
    }

    void step(GPU_ParticleSystem* ps, scalar dt) override;

    scalar _global_damping;
    ~GPU_PBD() override = default;
};

__global__ void kernel_velocity_update(int n, float dt, scalar global_damping, GPU_ParticleSystem_Parameters ps);
