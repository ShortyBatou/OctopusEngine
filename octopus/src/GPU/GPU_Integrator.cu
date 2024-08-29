#include "GPU/GPU_Integrator.h"

__global__ void kenerl_semi_exicit_integration(const int n, const scalar dt, const Vector3 g, Vector3 *p,
                                               Vector3 *prev_p, Vector3 *v, Vector3 *f, scalar *w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    prev_p[i] = p[i];
    v[i] += (g + f[i] * w[i]) * dt;
    p[i] += v[i] * dt;
    f[i] *= 0;
}


void GPU_SemiExplicit::integrate(GPU_ParticleSystem *ps, scalar dt) {
    int n = ps->nb();
    kenerl_semi_exicit_integration<<<n / 256, 256>>>(n, dt, Dynamic::gravity(),
                                                     ps->cb_position->buffer, ps->cb_prev_position->buffer,
                                                     ps->cb_velocity->buffer, ps->cb_forces->buffer,
                                                     ps->cb_inv_mass->buffer
    );
}
