#include "GPU/VBD/GPU_MG_VBD.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>


__global__ void kernel_mg_integration(
        const scalar dt, const Vector3 g,
        GPU_ParticleSystem_Parameters ps,
        Vector3* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    ps.last_p[i] = ps.p[i]; // x^t-1 = x^t
    const Vector3 a_ext = g + ps.f[i] * ps.w[i];
    y[i] = ps.p[i] + (ps.v[i] + a_ext * dt) * dt;
    ps.p[i] = y[i];
    ps.f[i] *= 0;
}

void GPU_MG_VBD:: step(const scalar dt) {
    const int n = nb_particles();
    // integration / first guess
    kernel_mg_integration<<<(n + 31)/32, 32>>>(dt,Dynamic::gravity(),
        get_parameters(),y->buffer);

    for(int j = 0; j < iteration; ++j) {
        // solve
        for(GPU_Dynamic* dynamic : _dynamics)
            dynamic->step(this, dt);

        for(GPU_Dynamic * constraint : _constraints)
            constraint->step(this, dt);
    }
    // velocity update
    kernel_velocity_update<<<(n + 31)/32, 32>>>(dt,get_parameters());

}