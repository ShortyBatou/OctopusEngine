#pragma once
#include "Core/Base.h"

class GPU_ParticleSystem;

struct GPU_Dynamic {
    GPU_Dynamic() : active(true) {}
    virtual ~GPU_Dynamic() = default;
    virtual void step(const GPU_ParticleSystem *ps, scalar dt) = 0;
    bool active;
};