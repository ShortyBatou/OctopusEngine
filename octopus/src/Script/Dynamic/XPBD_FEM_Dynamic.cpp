#include "Script/Dynamic/XPBD_FEM_Dynamic.h"

#include <Dynamic/PBD/PBD_ContinuousMaterial.h>
#include <Dynamic/PBD/XPBD_FEM_Generic.h>

void XPBD_FEM_Dynamic::set_iterations(int it, int sub_it) {
    _iteration = it;
    _sub_iteration = sub_it;
    PBD_System *pbd = dynamic_cast<PBD_System *>(this->_ps);
    pbd->_nb_step = it;
    pbd->_nb_substep = sub_it;
}

ParticleSystem *XPBD_FEM_Dynamic::build_particle_system() {
    return new PBD_System(new EulerSemiExplicit(1.f), _iteration, _sub_iteration, _type, _global_damping);
}

std::vector<FEM_Generic *> XPBD_FEM_Dynamic::build_element(const std::vector<int> &ids, const Element type, scalar &volume) {
    std::vector<PBD_ContinuousMaterial *> materials = get_pbd_materials(_material, _young, _poisson);
    PBD_System *pbd = dynamic_cast<PBD_System *>(_ps);
    XPBD_FEM_Generic *fem = nullptr;
    std::vector<FEM_Generic *> fems;
    for (PBD_ContinuousMaterial *m: materials) {
        fem = new XPBD_FEM_Generic(ids, m, get_fem_shape(type));
        fems.push_back(fem);
        pbd->add_xpbd_constraint(fem);
    }
    volume = fem->compute_volume(fem->get_particles(_ps->particles()));
    return fems;
}
