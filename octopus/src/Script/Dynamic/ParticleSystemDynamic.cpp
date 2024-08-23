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
