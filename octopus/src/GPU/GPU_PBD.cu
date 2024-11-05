#include "GPU/GPU_PBD.h"

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
    if(Input::Down(Key::A)) ++iteration;
    if(Input::Down(Key::Q)) --iteration;
    if(Input::Down(Key::A) || Input::Down(Key::Q)) std::cout << iteration << std::endl;
    const scalar sub_dt = dt / static_cast<scalar>(iteration);

    for(int i = 0; i < iteration; ++i) {

        integrator->integrate(this, sub_dt);
        for(auto* c : dynamic) {
            if(c->active)
                c->step(this, sub_dt);
        }

        kernel_velocity_update<<<(cb_position->nb + 255) / 256, 256>>>(cb_position->nb, sub_dt, global_damping,
                                                                   cb_position->buffer, cb_prev_position->buffer, cb_inv_mass->buffer, cb_velocity->buffer);
    }
}