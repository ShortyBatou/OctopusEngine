#include "GPU/PBD/GPU_PBD.h"

#include <Manager/Debug.h>
#include <Manager/Input.h>

// global device function
__global__ void kernel_velocity_update(const int n, const float dt, const scalar global_damping, Vector3 *p, Vector3 *prev_p, scalar* inv_mass, Vector3 *v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = (p[i] - prev_p[i]) / dt;
    scalar norm_v = glm::length(v[i]);
    if (norm_v > 1e-12) {
        const scalar coef = global_damping * dt * inv_mass[i];
        const scalar damping = -norm_v * (coef > 1.f ? 1.f : coef);
        v[i] += glm::normalize(v[i]) * damping;
    }
}

void GPU_PBD::step(const scalar dt)  {
    _integrator->step(this, dt);
    for(GPU_Dynamic* dynamic : _dynamics) {
        if(dynamic->active)
            dynamic->step(this, dt);
    }

    kernel_velocity_update<<<(_nb_particles + 255) / 256, 256>>>(_nb_particles, dt, _global_damping,
                                                               buffer_position(), buffer_prev_position(), buffer_inv_mass(), buffer_velocity());

}