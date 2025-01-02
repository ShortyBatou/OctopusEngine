#include "GPU/GPU_Integrator.h"

#include <GPU/CUMatrix.h>
#include <GPU/GPU_ParticleSystem.h>
#include <Manager/Dynamic.h>

__global__ void kenerl_semi_exicit_integration(const int n, const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ps.last_p[i] = ps.p[i];
    ps.v[i] += (g + ps.f[i] * ps.w[i]) * dt;
    ps.p[i] += ps.v[i] * dt;
    ps.f[i] *= 0;
}


void GPU_SemiExplicit::step(GPU_ParticleSystem *ps, const scalar dt) {
    const int n = ps->nb_particles();
    kenerl_semi_exicit_integration<<<(n+31) / 32, 32>>>(n, dt, Dynamic::gravity(), ps->get_parameters());
}
