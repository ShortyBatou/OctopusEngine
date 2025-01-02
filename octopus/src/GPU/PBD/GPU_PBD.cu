#include "GPU/PBD/GPU_PBD.h"

#include <Manager/Debug.h>
#include <Manager/Input.h>

// global device function
__global__ void kernel_velocity_update(const int n, const float dt, const scalar global_damping, GPU_ParticleSystem_Parameters ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ps.v[i] = (ps.p[i] - ps.last_p[i]) / dt;
    scalar norm_v = glm::length(ps.v[i]);
    if (norm_v > 1e-12) {
        const scalar coef = global_damping * dt * ps.w[i];
        const scalar damping = -norm_v * (coef > 1.f ? 1.f : coef);
        ps.v[i] += glm::normalize(ps.v[i]) * damping;
    }
}

void GPU_PBD::step(const scalar dt)  {
    _integrator->step(this, dt);
    for(GPU_Dynamic* dynamic : _dynamics) {
        if(dynamic->active)
            dynamic->step(this, dt);
    }
    for(GPU_Dynamic* constraint : _constraints) {
        if(constraint->active)
            constraint->step(this, dt);
    }

    kernel_velocity_update<<<(_data->_nb_particles + 255) / 256, 256>>>(_data->_nb_particles, dt, _global_damping, get_parameters());

}