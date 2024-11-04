#include "Script/Dynamic/XPBD_ShapeMatching_Dynamic.h"
#include "Manager/TimeManager.h"
#include <Dynamic/PBD/PBD_ContinuousMaterial.h>
#include <Dynamic/PBD/XPBD_ShapeMatching.h>

void XPBD_ShapeMatching_Dynamic::update() {
    this->_ps->step(Time::Fixed_DeltaTime());
    for (int i = 0; i < _mesh->nb_vertices(); ++i) {
        _mesh->geometry()[i] = this->_ps->get(i)->position;
    }
}


ParticleSystem *XPBD_ShapeMatching_Dynamic::build_particle_system() {
    return new PBD_System(new EulerSemiExplicit(1.f), _iteration, _sub_iteration, GaussSeidel, _global_damping);
}

void XPBD_ShapeMatching_Dynamic::build_dynamic() {
    for (Particle *p: _ps->particles()) {
        p->mass = 0;
    }
    auto *pbd = dynamic_cast<PBD_System *>(_ps);

    scalar total_volume = 0.f;
    for (auto &[type, topology]: _mesh->topologies()) {
        const int nb = elem_nb_vertices(type);
        const FEM_Shape* shape = get_fem_shape(type);

        std::vector<int> ids(nb);
        for (int i = 0; i < topology.size(); i += nb) {
            for (int j = 0; j < nb; ++j) {
                ids[j] = topology[i + j];
            }

            const scalar volume = FEM_Generic::compute_volume(shape, _ps->particles(), ids);
            total_volume += volume;
            for (int j = 0; j < nb; ++j) {
                Particle *p = _ps->get(ids[j]);
                p->mass += _density * volume / static_cast<scalar>(nb);
            }
        }
        delete shape;
    }

    scalar total_mass = 0.f;
    const int n = _ps->nb_particles();
    for (Particle *p: _ps->particles()) {
        //p->mass = total_volume * _density / static_cast<scalar>(n);
        p->inv_mass = 1.f / p->mass;
        total_mass += p->mass;
    }

    for (auto &[type, topology]: _mesh->topologies()) {
        const int nb = elem_nb_vertices(type);
        const FEM_Shape* shape = get_fem_shape(type);

        std::vector<int> ids(nb);
        for (int i = 0; i < topology.size(); i += nb) {
            for (int j = 0; j < nb; ++j) {
                ids[j] = topology[i + j];
            }

            const scalar volume = FEM_Generic::compute_volume(shape, _ps->particles(), ids);
            const std::vector<PBD_ContinuousMaterial *> materials = get_pbd_materials(_material, _young, _poisson);

            XPBD_ShapeMatching* sm = nullptr;
            for (PBD_ContinuousMaterial *m: materials) {
                sm = new XPBD_ShapeMatching(ids, m, volume);
                pbd->add_xpbd_constraint(sm);
            }
            pbd->add_xpbd_constraint(new XPBD_ShapeMatching_Filtering(sm));
            pbd->new_group();
        }
        delete shape;
    }
    pbd->shuffle_groups();

}