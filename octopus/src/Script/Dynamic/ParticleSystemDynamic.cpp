#include "Script/Dynamic/ParticleSystemDynamic.h"
#include "Core/Entity.h"

void ParticleSystemDynamic::init() {
    _mesh = this->entity()->get_component<Mesh>();
    _ps = build_particle_system();
    build_particles();
    build_dynamic();
}

void ParticleSystemDynamic::update_mesh() {
    for (int i = 0; i < this->_mesh->nb_vertices(); ++i) {
        _mesh->geometry()[i] = _ps->get(i)->position;
    }
}

void ParticleSystemDynamic::build_particles() {
    for (int i = 0; i < _mesh->nb_vertices(); ++i) {
        Vector3 p = Vector3(_mesh->geometry()[i]);
        _ps->add_particle(p, _particle_mass / static_cast<scalar>(_mesh->nb_vertices()));
    }
}


std::vector<Vector3> ParticleSystemDynamic::get_positions()
{
    std::vector<Vector3> positions(_ps->nb_particles());
    for (int i = 0; i < _ps->nb_particles(); ++i)
        positions[i] = _ps->get(i)->position;
    return positions;
}

std::vector<Vector3> ParticleSystemDynamic::get_init_positions()
{
    std::vector<Vector3> init_positions(_ps->nb_particles());
    for (int i = 0; i < _ps->nb_particles(); ++i)
        init_positions[i] = _ps->get(i)->init_position;
    return init_positions;
}

std::vector<Vector3> ParticleSystemDynamic::get_displacement()
{
    std::vector<Vector3> displascements(_ps->nb_particles());
    for (int i = 0; i < _ps->nb_particles(); ++i)
        displascements[i] = _ps->get(i)->position - _ps->get(i)->init_position;
    return displascements;
}

std::vector<Vector3> ParticleSystemDynamic::get_velocity()
{
    std::vector<Vector3> velocities(_ps->nb_particles());
    for (int i = 0; i < _ps->nb_particles(); ++i)
        velocities[i] = _ps->get(i)->velocity;
    return velocities;
}


std::vector<scalar> ParticleSystemDynamic::get_masses()
{
    std::vector<scalar> masses(_ps->nb_particles());
    for (int i = 0; i < _ps->nb_particles(); ++i)
        masses[i] = _ps->get(i)->mass;
    return masses;
}

std::vector<scalar> ParticleSystemDynamic::get_massses_inv()
{
    std::vector<scalar> inv_masses(_ps->nb_particles());
    for (int i = 0; i < _ps->nb_particles(); ++i)
        inv_masses[i] = _ps->get(i)->inv_mass;
    return inv_masses;
}

std::vector<scalar> ParticleSystemDynamic::get_displacement_norm()
{
    const std::vector<Vector3> displacement = get_displacement();
    std::vector<scalar> displacement_norm(displacement.size());
    for (int i = 0; i < displacement.size(); i++)
        displacement_norm[i] = glm::length(displacement[i]);

    return displacement_norm;
}

std::vector<scalar> ParticleSystemDynamic::get_velocity_norm()
{
    const std::vector<Vector3> velocity = get_velocity();
    std::vector<scalar> velocity_norm(velocity.size());
    for (int i = 0; i < velocity.size(); i++)
        velocity_norm[i] = glm::length(velocity[i]);
    return velocity_norm;
}