#pragma once
#include <random>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"

struct VBD_Object {
    virtual ~VBD_Object() = default;
    virtual void solve(ParticleSystem* ps, scalar dt) = 0;
    virtual void compute_inertia(ParticleSystem *ps, scalar dt) = 0;
};
