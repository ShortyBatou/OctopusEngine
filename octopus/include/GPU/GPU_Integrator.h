#pragma once
#include "Core/Base.h"
#include "GPU/GPU_ParticleSystem.h"
struct GPU_Integrator {
    virtual ~GPU_Integrator() = default;
    virtual void integrate(const GPU_ParticleSystem* ps, scalar dt) = 0;
};

struct GPU_SemiExplicit final : GPU_Integrator {
    void integrate(const GPU_ParticleSystem* ps, scalar dt) override;
};
