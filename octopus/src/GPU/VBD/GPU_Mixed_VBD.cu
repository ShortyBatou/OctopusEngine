#include "GPU/VBD/GPU_Mixed_VBD.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>

__global__ void kenerl_semi_exicit_integration2(const int n, const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps, scalar* w_max) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ps.mask[i] == 0) return;
    Vector3 dt_v = ps.f[i] * ps.w[i] * dt;
    scalar dt_limit = 0.75f / sqrtf(w_max[i]);
    if(dt < dt_limit || ps.mask[i] == 2) // 0.75 is supposed to be equal to 2, its a magic number
    {
        ps.mask[i] = 2;
        dt_v = Vector3(0);
    }
    ps.v[i] += g * dt + dt_v;
    ps.p[i] += ps.v[i] * dt;
    ps.f[i] *= 0;
}

__global__ void kernel_reset_mask(const int n, GPU_ParticleSystem_Parameters ps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ps.mask[i] == 0) return;
    ps.mask[i] = 1;
}

__global__ void kenerl_semi_exicit_integration3(const int n, const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ps.mask[i] == 0) return;
    Vector3 dt_v = ps.f[i] * ps.w[i] * dt;
    if(glm::length(dt_v) > 10 || ps.mask[i] == 2) // 0.75 is supposed to be equal to 2, its a magic number
    {
        ps.mask[i] = 2;
        dt_v = Vector3(0);
    }
    ps.v[i] += g * dt + dt_v;
    ps.p[i] += ps.v[i] * dt;
    ps.f[i] *= 0;
}


__global__ void kernel_inertia(const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps, Vector3* y, Vector3* prev_it_p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    ps.last_p[i] = ps.p[i]; // x^t-1 = x^t
    prev_it_p[i] = ps.p[i];
    const Vector3 a_ext = g + ps.f[i] * ps.w[i];
    y[i] = ps.p[i] + (ps.v[i] + a_ext * dt) * dt;
}

void GPU_Mixed_VBD::step(const scalar dt) {
    const int n = nb_particles();

    // Compute inertia and save last_p (doesn't change v or p nor f)
    kernel_inertia<<<(n + 31)/32, 32>>>(dt,Dynamic::gravity(), get_parameters(),y->buffer, prev_it_p->buffer);

    const scalar dt_exp = dt / static_cast<scalar>(explicit_it);
    for(int i = 0; i < explicit_it; ++i)
    {
        // eval forces
        for(const GPU_Mixed_VBD_FEM* fem : _fems)
            fem->explicit_step(this, w_max, dt_exp);

        // integrations
        //kenerl_semi_exicit_integration2<<<(n+31) / 32, 32>>>(n, dt_exp, Dynamic::gravity(), get_parameters(), w_max->buffer);
        kenerl_semi_exicit_integration3<<<(n+31) / 32, 32>>>(n, dt_exp, Dynamic::gravity(), get_parameters());

        for(GPU_Dynamic * constraint : _constraints)
            constraint->step(this, dt_exp);
    }
    kernel_reset_mask<<<(n+31) / 32, 32>>>(n, get_parameters());

    scalar omega = 1;
    for(int j = 0; j < iteration; ++j) {
        // solve
        for(GPU_Dynamic* dynamic : _dynamics)
            dynamic->step(this, dt);

        for(GPU_Dynamic * constraint : _constraints)
            constraint->step(this, dt);

        // Acceleration (Chebychev)
        if(j == 1) omega = 2.f / (2.f - _rho * _rho);
        else if(j > 1) omega = 4.f / (4.f - _rho * _rho * omega);
        //Skernel_chebychev_acceleration<<<(n + 255)/256, 256>>>(j, omega, get_parameters(), prev_it_p->buffer, prev_it2_p->buffer);
    }

    // velocity update
    kernel_velocity_update<<<(n + 255)/256, 256>>>(dt,get_parameters());

}