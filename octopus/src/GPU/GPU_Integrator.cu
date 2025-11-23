#include "GPU/GPU_Integrator.h"

#include <GPU/CUMatrix.h>
#include <GPU/GPU_ParticleSystem.h>
#include <Manager/Dynamic.h>
#include <algorithm>

void GPU_Integrator::init_dynamics(GPU_ParticleSystem *ps, scalar dt)
{
    for(GPU_Dynamic* dynamic : _dynamics)
        if(dynamic->active) dynamic->start(ps, dt);
}

void GPU_Integrator::eval_dynamics(GPU_ParticleSystem *ps, scalar dt)
{
    for(GPU_Dynamic* dynamic : _dynamics)
        if(dynamic->active) dynamic->step(ps, dt);
}

void GPU_Integrator::eval_constraints(GPU_ParticleSystem *ps, scalar dt)
{
    for(GPU_Dynamic* constraint : _constraints)
        if(constraint->active) constraint->step(ps, dt);
}


__global__ void kenerl_semi_exicit_integration(const int n, const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ps.last_p[i] = ps.p[i];
    ps.v[i] += (g + ps.f[i] * ps.w[i]) * dt;
    ps.p[i] += ps.v[i] * dt;
    ps.f[i] *= 0;
}

void GPU_SemiExplicit::step(GPU_ParticleSystem *ps, const scalar dt) {

    eval_dynamics(ps, dt);
    const int n = ps->nb_particles();
    kenerl_semi_exicit_integration<<<(n+31) / 32, 32>>>(n, dt, Dynamic::gravity(), ps->get_parameters());
    if(apply_constraints) eval_constraints(ps, dt);
}


__global__ void kernel_explicit_rk(const int n, const scalar dt, const Vector3 g, int order, GPU_ParticleSystem_Parameters ps, RK_Data k) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    if(k.step == 0)
    {
        ps.last_p[id] = ps.p[id];
    }
    if(k.step == 0 && k.step < order -1)
    {
        k.v[id] = ps.v[id];
        k.x[id] = ps.p[id];
        k.dv[id] = Vector3(0,0,0);
        k.dx[id] = Vector3(0,0,0);
    }

    Vector3 k_dv = g + ps.f[id] * ps.w[id];
    Vector3 k_dx = ps.v[id];

    if(k.step < order-1)
    {
        k.dv[id] += k.w2 * (g + ps.f[id] * ps.w[id]);
        k.dx[id] += k.w2 * ps.v[id];
    }
    else
    {
        k_dv = k.dv[id] + k.w2 * (g + ps.f[id] * ps.w[id]);
        k_dx = k.dx[id] + k.w2 * (ps.v[id]);
    }

    ps.v[id] = k.v[id] + k.w1 * k_dv * dt;
    ps.p[id] = k.x[id] + k.w1 * k_dx * dt;
    ps.f[id] *= 0;
}


void GPU_Explicit_RK::step(GPU_ParticleSystem* ps, scalar dt)
{

    RK_Data data = {
        cb_x->buffer, cb_v->buffer, 0,0,0, cb_kv->buffer, cb_kx->buffer
    };

    const int n = ps->nb_particles();
    for(int i = 0; i < order; ++i)
    {
        data.w1 = w1[i]; data.w2 = w2[i]; data.step = i;
        eval_dynamics(ps, dt * w2[i]);
        kernel_explicit_rk<<<(n+31)/32,32>>>(n, dt, Dynamic::gravity(), order, ps->get_parameters(), data);
    }

    if(apply_constraints) eval_constraints(ps,dt);
}

