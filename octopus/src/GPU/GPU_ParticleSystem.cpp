#include "GPU/GPU_ParticleSystem.h"

GPU_ParticleSystem::GPU_ParticleSystem(const std::vector<Vector3>& positions, const std::vector<scalar>& masses) {
    n = static_cast<int>(positions.size());
    std::vector inv_mass(masses);
    std::for_each(inv_mass.begin(), inv_mass.end(), [](scalar &n){ n = 1.f / n; });

    cb_position = new Cuda_Buffer(positions);
    cb_prev_position = new Cuda_Buffer(positions);
    cb_init_position = new Cuda_Buffer(positions);
    cb_velocity = new Cuda_Buffer(std::vector(n, Unit3D::Zero()));
    cb_forces = new Cuda_Buffer(std::vector(n, Unit3D::Zero()));
    cb_mass = new Cuda_Buffer(masses);
    cb_inv_mass = new Cuda_Buffer(inv_mass);
}