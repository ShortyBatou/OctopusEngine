#include "GPU/GPU_Integrator.h"

#include <GPU/CUMatrix.h>
#include <GPU/GPU_ParticleSystem.h>
#include <Manager/Dynamic.h>
#include <algorithm>

__global__ void kenerl_semi_exicit_integration(const int n, const scalar dt, const Vector3 g, GPU_ParticleSystem_Parameters ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ps.last_p[i] = ps.p[i];
    ps.v[i] += (g + ps.f[i] * ps.w[i]) * dt;
    ps.p[i] += ps.v[i] * dt;
    ps.f[i] *= 0;
}


GPU_ParticleSystem::GPU_ParticleSystem(const std::vector<Vector3>& positions, const std::vector<scalar>& masses,
                                       GPU_Integrator* integrator, int sub_iteration)
        : _sub_iteration(sub_iteration), _integrator(integrator)
{
    const int nb_particle = static_cast<int>(positions.size());
    _data = new GPU_ParticleSystem_Data();
    _data->_nb_particles = nb_particle;
    std::vector inv_mass(masses);
    std::for_each(inv_mass.begin(), inv_mass.end(), [](scalar& n) { n = 1.f / n; });

    _data->_cb_position = new Cuda_Buffer(positions);
    _data->_cb_prev_position = new Cuda_Buffer(positions);
    _data->_cb_init_position = new Cuda_Buffer(positions);
    _data->_cb_velocity = new Cuda_Buffer(std::vector(nb_particle, Unit3D::Zero()));
    _data->_cb_forces = new Cuda_Buffer(std::vector(nb_particle, Unit3D::Zero()));
    _data->_cb_mass = new Cuda_Buffer(masses);
    _data->_cb_inv_mass = new Cuda_Buffer(inv_mass);
    _data->_cb_mask = new Cuda_Buffer(std::vector(nb_particle, 1));
}

GPU_ParticleSystem_Parameters GPU_ParticleSystem::get_parameters() const
{
    GPU_ParticleSystem_Parameters params{};
    params.nb_particles = _data->_nb_particles;
    params.f = _data->_cb_forces->buffer;
    params.init_p = _data->_cb_init_position->buffer;
    params.w =_data-> _cb_inv_mass->buffer;
    params.mask = _data->_cb_mask->buffer;
    params.m = _data->_cb_mass->buffer;
    params.p = _data->_cb_position->buffer;
    params.last_p = _data->_cb_prev_position->buffer;
    params.v = _data->_cb_velocity->buffer;
    return params;
}


void GPU_ParticleSystem::step(const scalar dt)
{
    for(GPU_Dynamic* dynamic : _dynamics) dynamic->step(this, dt);
    _integrator->step(this, dt);
    for(GPU_Dynamic* constraint : _constraints) constraint->step(this, dt);

}

void GPU_SemiExplicit::step(GPU_ParticleSystem *ps, const scalar dt) {
    const int n = ps->nb_particles();
    kenerl_semi_exicit_integration<<<(n+31) / 32, 32>>>(n, dt, Dynamic::gravity(), ps->get_parameters());
}
