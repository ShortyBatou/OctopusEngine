#include "GPU/GPU_VBD.h"
#include "GPU/CUMatrix.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include <Manager/TimeManager.h>


__global__ void kernel_integration(
        const int n, const scalar dt, const Vector3 g,
        Vector3 *p, Vector3 *prev_p, Vector3* y, Vector3* prev_it_p, Vector3 *v, Vector3 *f, scalar *w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    prev_p[i] = p[i]; // x^t-1 = x^t
    prev_it_p[i] = p[i];
    const Vector3 a_ext = g + f[i] * w[i];
    y[i] = p[i] + (v[i] + a_ext * dt) * dt;
    p[i] = y[i];
    f[i] *= 0;
}

__global__ void kernel_velocity_update(int n, scalar dt, Vector3* prev_p, Vector3* p, Vector3* v, scalar* _inv_mass) {
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= n) return;
    v[vid] = (p[vid] - prev_p[vid]) / dt;
    //if(vid == 10) printf("v(%f %f %f)\n", v[vid].x, v[vid].y, v[vid].z);
}

__global__ void kernel_chebychev_acceleration(int n, int it, scalar omega, Vector3* prev_it_p, Vector3* prev_it2_p, Vector3* p) {
    const int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= n) return;
    if(it >= 2) {
        p[vid] = prev_it2_p[vid] + omega * (p[vid] - prev_it2_p[vid]);
    }
    prev_it2_p[vid] = prev_it_p[vid];
    prev_it_p[vid] = p[vid];
}


void GPU_VBD::step(const scalar dt) {
    const int n = nb_particles();

    //scalar omega = 1;
    // integration / first guess
    kernel_integration<<<(n + 255)/256, 256>>>(n,dt,Dynamic::gravity(),
        buffer_position(),buffer_prev_position(),y->buffer, prev_it_p->buffer,
        buffer_velocity(),buffer_forces(), buffer_inv_mass());

    for(int j = 0; j < iteration; ++j) {
        const scalar r = 0.8f;
        // solve
        for(GPU_Dynamic* dynamic : _dynamics)
            dynamic->step(this, dt);

        for(GPU_Dynamic * constraint : _constraints)
            constraint->step(this, dt);

        // Acceleration (Chebychev)
        //if(j == 1) omega = 2.f / (2.f - r * r);
        //else if(j > 1) omega = 4.f / (4.f - r * r * omega);
        //kernel_chebychev_acceleration<<<(n + 255)/256, 256>>>(n, j, omega, prev_it_p->buffer, prev_it2_p->buffer, cb_position->buffer);
    }
    // velocity update
    kernel_velocity_update<<<(n + 255)/256, 256>>>(n,dt,
        buffer_prev_position(), buffer_position(), buffer_velocity(), buffer_inv_mass());

}

GPU_VBD::~GPU_VBD()
{

}