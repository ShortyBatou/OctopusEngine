#pragma once
#include "GPU_Dynamic.h"
#include "Core/Base.h"

class ParticleSystem;

struct GPU_Integrator : GPU_Dynamic {
    ~GPU_Integrator() override = default;
    void step(GPU_ParticleSystem* ps, scalar dt) override = 0;
};

struct GPU_SemiExplicit final : GPU_Integrator {
    void step(GPU_ParticleSystem* ps, scalar dt) override;
};
