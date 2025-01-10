#include "GPU/VBD/GPU_Mixed_VBD.h"
#include <glm/detail/func_matrix_simd.inl>
#include <GPU/CUMatrix.h>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>

// ULTRA damped, don't know why
__global__ void kernel_rk4(
    const int n, scalar dt, const Vector3 g, const int step,
    GPU_ParticleSystem_Parameters ps, Vector3* l, Vector3* k, Vector3* rk4_last_p)
{
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n || ps.mask[vid] == 0) return;
    if(step == 0) {
        rk4_last_p[vid] = ps.p[vid];
    }
    const int i = vid * 4 + step;

    k[i] = ps.v[vid] + (step == 0 ? Vector3(0.) : 0.5f * dt * l[i-1]); // velocity
    l[i] = ps.f[vid] * ps.w[vid] + g; // acceleration

    if(step == 3) {
        Vector3 dt_p = (dt/6.f) * (k[vid * 4] + 2.f * k[vid*4+1] + 2.f * k[vid*4+2] + k[vid*4+3]);
        Vector3 dt_v = (dt/6.f) * (l[vid * 4] + 2.f * l[vid*4+1] + 2.f * l[vid*4+2] + l[vid*4+3]);
        if(glm::length(dt_v) > 0.1f) {
            dt_v = g*dt;
            dt_p = dt_v * dt;
        }
        ps.v[vid] = ps.v[vid]    + dt_v;
        ps.p[vid] = rk4_last_p[vid] + dt_p;
    } else {
        ps.p[vid] = rk4_last_p[vid] + k[i] * dt * 0.5f;
    }
    ps.f[vid] *= 0;
}

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

__global__ void kenerl_semi_exicit_integration3(const int n, const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps, const Vector3* last_v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || ps.mask[i] == 0) return;
    const Vector3 dt_v = ps.f[i] * ps.w[i] * dt;
    if(glm::length(dt_v) > 10 && ps.mask[i] != 2) // 0.75 is supposed to be equal to 2, its a magic number
    {
        ps.mask[i] = 2;
        ps.v[i] = last_v[i];
    }
    ps.v[i] += g * dt + (ps.mask[i] == 2 ? Vector3(0.f) : dt_v);
    ps.p[i] += ps.v[i] * dt;
    ps.f[i] *= 0;
}


__global__ void kernel_inertia(const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps, Vector3* y, Vector3* last_v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    ps.last_p[i] = ps.p[i]; // x^t-1 = x^t
    const Vector3 a_ext = g + ps.f[i] * ps.w[i];
    const Vector3 dt_v = ps.v[i] + a_ext * dt;
    last_v[i] = dt_v;
    y[i] = ps.p[i] + dt_v * dt;
}

void GPU_Mixed_VBD::step(const scalar dt) {
    const int n = nb_particles();
    GPU_ParticleSystem_Parameters ps_param = get_parameters();
    // Compute inertia and save last_p (doesn't change v or p nor f)
    kernel_inertia<<<(n + 31)/32, 32>>>(dt,Dynamic::gravity(), ps_param,y->buffer, last_v->buffer);

    const scalar dt_exp = dt / static_cast<scalar>(explicit_it);
    for(int i = 0; i < explicit_it; ++i)
    {
        // eval forces
        // integrations Euler semi-implicit
        //for(const GPU_Mixed_VBD_FEM* fem : _fems) fem->explicit_step(this, w_max, dt_exp);
        //kenerl_semi_exicit_integration2<<<(n+31) / 32, 32>>>(n, dt_exp, Dynamic::gravity(), get_parameters(), w_max->buffer);/**/
        //kenerl_semi_exicit_integration3<<<(n+31) / 32, 32>>>(n, dt_exp, Dynamic::gravity(), get_parameters(), last_v->buffer);/**/

        // integration Runge-Kutta
        for(int j = 0; j < 4; ++j) {
            for(const GPU_Mixed_VBD_FEM* fem : _fems)
                fem->explicit_step(this, w_max, dt_exp);
            kernel_rk4<<<(n+31) / 32, 32>>>(n, dt_exp, Dynamic::gravity(), j, ps_param, l->buffer, k->buffer, rk4_last_p->buffer);
        }/**/
    }
    kernel_reset_mask<<<(n+31) / 32, 32>>>(n, ps_param);

    for(int j = 0; j < iteration; ++j) {
        // solve
        for(GPU_Dynamic* dynamic : _dynamics)
            dynamic->step(this, dt);

        for(GPU_Dynamic * constraint : _constraints)
            constraint->step(this, dt);
    }

    // velocity update
    kernel_velocity_update<<<(n + 255)/256, 256>>>(dt,ps_param);

}