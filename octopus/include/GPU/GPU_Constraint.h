#pragma once
#include "Core/Base.h"
#include <Mesh/Mesh.h>
#include <Tools/Area.h>

#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_Fix_Constraint final : GPU_Dynamic {
    GPU_Fix_Constraint(const std::vector<Vector3>& positions, Area* arean);
    void step(GPU_ParticleSystem *ps, scalar dt) override;
    ~GPU_Fix_Constraint() override { delete cb_ids;}
    Axis axis;
    Vector3 com;
    Cuda_Buffer<int>* cb_ids;
};

struct GPU_Box_Limit final : GPU_Dynamic {
    GPU_Box_Limit(Vector3 _pmin, Vector3 _pmax) : pmin(_pmin), pmax(_pmax) {}
    void step(GPU_ParticleSystem *ps, scalar dt) override;
    Vector3 pmin, pmax;
};


struct GPU_Crush final : GPU_Dynamic {
    void step(GPU_ParticleSystem *ps, scalar dt) override;
};


struct GPU_RandomSphere final : GPU_Dynamic {
    GPU_RandomSphere(const Vector3 c, const scalar r) : radius(r), center(c) {}
    void step(GPU_ParticleSystem *ps, scalar dt) override;
    scalar radius;
    Vector3 center;
};
