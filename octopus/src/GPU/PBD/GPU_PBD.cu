#include "GPU/PBD/GPU_PBD.h"

#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include <Manager/Input.h>

// global device function
__global__ void kernel_velocity_update(const int n, const float dt, const scalar global_damping, GPU_ParticleSystem_Parameters ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ps.v[i] = (ps.p[i] - ps.last_p[i]) / dt;
    scalar norm_v = glm::length(ps.v[i]);
    if (norm_v > 1e-7) {
        const scalar coef = global_damping * dt * ps.w[i];
        const scalar damping = -norm_v * (coef > 1.f ? 1.f : coef);
        ps.v[i] += glm::normalize(ps.v[i]) * damping;
    }
}

void GPU_PBD::step(GPU_ParticleSystem* ps, const scalar dt)  {
    int n = ps->nb_particles();
    kenerl_semi_exicit_integration<<<(n+31) / 32, 32>>>(n, dt, Dynamic::gravity(), ps->get_parameters());
    eval_dynamics(ps, dt);
    eval_constraints(ps, dt);
    kernel_velocity_update<<<(n + 31) / 32, 32>>>(n, dt, _global_damping, ps->get_parameters());
}