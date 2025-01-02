#pragma once
#include "Script/Dynamic/Cuda_ParticleSystem_Dynamic.h"

#include <Manager/TimeManager.h>


void Cuda_ParticleSystem_Dynamics::init()
{
    _mesh = this->entity()->get_component<Mesh>();
    _gpu_ps = create_particle_system();
    build_dynamics();
}

void Cuda_ParticleSystem_Dynamics::update()
{
    Time::Tic();
    const scalar sub_dt = Time::Fixed_DeltaTime() / static_cast<scalar>(_sub_iterations);
    for(int i = 0; i < _sub_iterations; i++)
        _gpu_ps->step(sub_dt);
    cudaDeviceSynchronize();
    _gpu_ps->get_position(_mesh->geometry());
    const scalar t = Time::Tac() * 1000.f;
    DebugUI::Begin("Entity Time " + std::to_string(entity()->id()));
    DebugUI::Plot("Time (" + std::to_string(entity()->id()) + ")", t);
    DebugUI::Range("Range (" + std::to_string(entity()->id()) + ")", t);
    DebugUI::Value("Value (" + std::to_string(entity()->id()) + ")", t);
    DebugUI::End();
}

std::vector<Vector3> Cuda_ParticleSystem_Dynamics::get_positions()
{
    std::vector<Vector3> positions;
    _gpu_ps->get_position(positions);
    return positions;
}

std::vector<Vector3> Cuda_ParticleSystem_Dynamics::get_init_positions()
{
    std::vector<Vector3> init_positions;
    _gpu_ps->get_init_position(init_positions);
    return init_positions;
}

std::vector<Vector3> Cuda_ParticleSystem_Dynamics::get_displacement()
{
    std::vector<Vector3> init_positions;
    std::vector<Vector3> position;
    _gpu_ps->get_init_position(init_positions);
    _gpu_ps->get_position(position);
    for(int i = 0; i < position.size(); ++i) position[i] -= init_positions[i];
    return position;
}

std::vector<Vector3> Cuda_ParticleSystem_Dynamics::get_velocity()
{
    std::vector<Vector3> velocity;
    _gpu_ps->get_velocity(velocity);
    return velocity;
}

std::vector<scalar> Cuda_ParticleSystem_Dynamics::get_masses()
{
    std::vector<scalar> masses;
    _gpu_ps->get_mass(masses);
    return masses;
}

std::vector<scalar> Cuda_ParticleSystem_Dynamics::get_massses_inv()
{
    std::vector<scalar> inv_masses;
    _gpu_ps->get_inv_mass(inv_masses);
    return inv_masses;
}

std::vector<scalar> Cuda_ParticleSystem_Dynamics::get_displacement_norm()
{
    const std::vector<Vector3> displacement = get_displacement();
    std::vector<scalar> norm(displacement.size());
    for(int i = 0; i < displacement.size(); ++i)
        norm[i] = glm::length(displacement[i]);

    return norm;
}

std::vector<scalar> Cuda_ParticleSystem_Dynamics::get_velocity_norm()
{
    std::vector<Vector3> velocity;
    _gpu_ps->get_velocity(velocity);
    std::vector<scalar> norm(velocity.size());
    for(int i = 0; i < velocity.size(); ++i)
        norm[i] = glm::length(velocity[i]);
    return norm;
}

