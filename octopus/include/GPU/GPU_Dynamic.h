#pragma once
#include "Core/Base.h"

struct GPU_ParticleSystem;

struct GPU_Dynamic {
    GPU_Dynamic() : active(true) {}
    virtual ~GPU_Dynamic() = default;
    virtual void start(GPU_ParticleSystem *ps, scalar dt) {} // 1 per loop
    virtual void step(GPU_ParticleSystem *ps, scalar dt) = 0; // 1 per iteration
    bool active;
};