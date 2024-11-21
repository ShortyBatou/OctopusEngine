#pragma once
#include <random>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"

struct VertexBlockDescent;

struct VBD_Object {
    virtual ~VBD_Object() = default;
    virtual void compute_inertia(VertexBlockDescent *ps, scalar dt) = 0;
    virtual void solve(VertexBlockDescent* vbd, scalar dt) = 0;
};
