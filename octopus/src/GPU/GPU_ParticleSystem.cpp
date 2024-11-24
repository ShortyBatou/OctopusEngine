#include "GPU/GPU_ParticleSystem.h"
#include <algorithm>

GPU_ParticleSystem::GPU_ParticleSystem(const std::vector<Vector3>& positions, const std::vector<scalar>& masses,
                                       GPU_Integrator* integrator, int sub_iteration)
        : _integrator(integrator), _sub_iteration(sub_iteration)
{
    _nb_particles = static_cast<int>(positions.size());
    std::vector inv_mass(masses);
    std::for_each(inv_mass.begin(), inv_mass.end(), [](scalar& n) { n = 1.f / n; });

    _cb_position = new Cuda_Buffer(positions);
    _cb_prev_position = new Cuda_Buffer(positions);
    _cb_init_position = new Cuda_Buffer(positions);
    _cb_velocity = new Cuda_Buffer(std::vector(_nb_particles, Unit3D::Zero()));
    _cb_forces = new Cuda_Buffer(std::vector(_nb_particles, Unit3D::Zero()));
    _cb_mass = new Cuda_Buffer(masses);
    _cb_inv_mass = new Cuda_Buffer(inv_mass);
    _cb_mask = new Cuda_Buffer(std::vector(_nb_particles, 1));
}

void GPU_ParticleSystem::step(scalar dt)
{
    for(GPU_Dynamic* dynamic : _dynamics) dynamic->step(this, dt);
    _integrator->step(this, dt);
}