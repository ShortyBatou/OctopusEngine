#pragma once
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/Cuda_Buffer.h"
#include "cuda_runtime.h"
#include "GPU_Integrator.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_Dynamic.h"


struct GPU_PBD : GPU_ParticleSystem {
    GPU_PBD(const std::vector<Vector3>& positions, const std::vector<scalar>& masses, const int it, const scalar damping = 0.f)
    : GPU_ParticleSystem(positions, masses), iteration(it), global_damping(damping), integrator(new GPU_SemiExplicit()) {}

    void step(scalar dt);

    int iteration;
    scalar global_damping;
    GPU_Integrator* integrator;
    std::vector<GPU_Dynamic*> dynamic;
    ~GPU_PBD() {
        delete integrator;
        for(const auto* d : dynamic) delete d;
    }
};

__global__ void kernel_velocity_update(int n, float dt, scalar global_damping, Vector3 *p, Vector3 *prev_p, scalar* inv_mass, Vector3 *v);
