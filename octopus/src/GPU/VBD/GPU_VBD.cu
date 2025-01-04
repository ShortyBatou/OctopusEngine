#include "GPU/VBD/GPU_VBD.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>

__global__ void kernel_integration(
        const scalar dt, const Vector3 g,
        GPU_ParticleSystem_Parameters ps,
        Vector3* y, Vector3* prev_it_p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    ps.last_p[i] = ps.p[i]; // x^t-1 = x^t
    prev_it_p[i] = ps.p[i];
    const Vector3 a_ext = g + ps.f[i] * ps.w[i];
    y[i] = ps.p[i] + (ps.v[i] + a_ext * dt) * dt;
    ps.p[i] = y[i];
    ps.f[i] *= 0;
}

__global__ void kernel_velocity_update(scalar dt, GPU_ParticleSystem_Parameters ps) {
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= ps.nb_particles) return;
    ps.v[vid] = (ps.p[vid] - ps.last_p[vid]) / dt;
    //if(vid == 10) printf("v(%f %f %f)\n", v[vid].x, v[vid].y, v[vid].z);
}

__global__ void kernel_chebychev_acceleration(const int it, const scalar omega, GPU_ParticleSystem_Parameters ps, Vector3* prev_it_p, Vector3* prev_it2_p) {
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= ps.nb_particles) return;
    if(it >= 2) {
        ps.p[vid] = prev_it2_p[vid] + omega * (ps.p[vid] - prev_it2_p[vid]);
    }
    prev_it2_p[vid] = prev_it_p[vid];
    prev_it_p[vid] = ps.p[vid];
}


void GPU_VBD::step(const scalar dt) {
    const int n = nb_particles();
    scalar omega = 1;
    // integration / first guess
    kernel_integration<<<(n + 31)/32, 32>>>(dt,Dynamic::gravity(),
        get_parameters(),y->buffer, prev_it_p->buffer);

    for(int j = 0; j < iteration; ++j) {
        // solve
        for(GPU_Dynamic* dynamic : _dynamics)
            dynamic->step(this, dt);

        for(GPU_Dynamic * constraint : _constraints)
            constraint->step(this, dt);

        // Acceleration (Chebychev)
        if(j == 1) omega = 2.f / (2.f - _rho * _rho);
        else if(j > 1) omega = 4.f / (4.f - _rho * _rho * omega);
        kernel_chebychev_acceleration<<<(n + 255)/256, 256>>>(j, omega, get_parameters(), prev_it_p->buffer, prev_it2_p->buffer);
    }
    // velocity update
    kernel_velocity_update<<<(n + 255)/256, 256>>>(dt,get_parameters());

}