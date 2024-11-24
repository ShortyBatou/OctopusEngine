#include "GPU/GPU_Integrator.h"

#include <GPU/CUMatrix.h>
#include <GPU/GPU_ParticleSystem.h>
#include <Manager/Dynamic.h>

__global__ void kenerl_semi_exicit_integration(const int n, const scalar dt, const Vector3 g, Vector3 *p,
                                               Vector3 *prev_p, Vector3 *v, Vector3 *f, scalar *w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if(i == 0) printf("========================== \n");
    prev_p[i] = p[i];
    v[i] += (g + f[i] * w[i]) * dt;
    print_vec(f[i]);
    p[i] += v[i] * dt;
    f[i] *= 0;
}


void GPU_SemiExplicit::step(const GPU_ParticleSystem *ps, const scalar dt) {
    const int n = ps->nb_particles();

    kenerl_semi_exicit_integration<<<(n+255) / 256, 256>>>(n, dt, Dynamic::gravity(),
                                                     ps->buffer_position(), ps->buffer_prev_position(),
                                                     ps->buffer_velocity(), ps->buffer_forces(),
                                                     ps->buffer_inv_mass()
    );
}
