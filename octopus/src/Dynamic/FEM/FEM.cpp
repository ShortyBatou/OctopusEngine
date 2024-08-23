#include "Dynamic/FEM/FEM.h"

void FEM_System::step(const scalar dt) {
    scalar h = dt / (scalar) _nb_substep;
    this->step_effects(dt);
    for (unsigned int i = 0; i < _nb_substep; i++) {
        this->step_force(h);
        this->step_solver(h);
        this->step_constraint(dt);
    }
}

FEM_System::~FEM_System() {
    clear_fem();
}

void FEM_System::clear_fem() {
    for (Constraint *c: _fems) delete c;
    _fems.clear();
}

int FEM_System::add_fem(Constraint *constraint) {
    _fems.push_back(constraint);
    _fems.back()->init(this->_particles);
    return static_cast<int>(_fems.size());
}

void FEM_System::step_force(const scalar dt) {
    for (Constraint *fem: _fems) {
        if (!fem->active()) continue;
        fem->apply(this->_particles, dt); // if xpbd
    }
}
