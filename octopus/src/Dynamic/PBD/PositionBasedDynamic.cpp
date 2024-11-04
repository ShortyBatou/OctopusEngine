#include "Dynamic/PBD/PositionBasedDynamic.h"
#include <algorithm>
#include <random>
#include <iterator>


void PBD_System::step(const scalar dt) {
    const scalar h = dt / static_cast<scalar>(_nb_substep);
    for (int i = 0; i < _nb_substep; i++) {
        this->step_solver(h);
        this->reset_lambda();
        for (int j = 0; j < _nb_step; ++j) {
            if (_type == Jacobi) {
                step_constraint_jacobi(h);
            }
            if (_type == GaussSeidel_RNG) {
                shuffle_groups();
                step_constraint_gauss_seidel(h);
            }
            else {
                step_constraint_gauss_seidel(h);
            }
        }

        this->step_constraint(h); // optional

        this->step_effects(h); // optional

        this->update_velocity(h);
    }
    this->reset_external_forces();
}

scalar PBD_System::get_residual(const scalar dt) const {
    const scalar sub_dt = dt / static_cast<scalar>(_nb_substep);
    scalar residual = 0;
    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        residual += xpbd->get_dual_residual(_particles, sub_dt);
    }
    return residual / static_cast<scalar>(_xpbd_constraints.size());
}

PBD_System::~PBD_System() {
    clear_xpbd_constraints();
}

void PBD_System::clear_xpbd_constraints() {
    for (const XPBD_Constraint *c: _xpbd_constraints) delete c;
    _xpbd_constraints.clear();
}

int PBD_System::add_xpbd_constraint(XPBD_Constraint *constraint) {
    _xpbd_constraints.push_back(constraint);
    _xpbd_constraints.back()->init(_particles);
    _groups[_groups.size() - 1].push_back(constraint);
    return static_cast<int>(_xpbd_constraints.size());
}

void PBD_System::new_group() {
    _groups.emplace_back();
}

void PBD_System::draw_debug_xpbd() const {
    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        xpbd->draw_debug(_particles);
    }
}


void PBD_System::reset_lambda() const {
    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        xpbd->set_lambda(0);
    }
}

void PBD_System::update_velocity(const scalar dt) const {
    for (Particle *p: this->_particles) {
        if (!p->active) continue;
        p->velocity = (p->position - p->last_position) / dt;

        const scalar norm_v = glm::length(p->velocity);
        if (norm_v > 1e-12) {
            const scalar damping = -norm_v * std::min(1.f, _global_damping * dt * p->inv_mass);
            p->velocity += glm::normalize(p->velocity) * damping;
        }
    }
}

void PBD_System::shuffle_groups() {
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(_groups), std::end(_groups), rng);
}

void PBD_System::step_constraint_gauss_seidel(const scalar dt) {
    for (std::vector<XPBD_Constraint *> &group: _groups) {
        for (XPBD_Constraint *xpbd: group) {
            if (!xpbd->active()) continue;

            // compute corrections
            xpbd->apply(_particles, dt);

            // apply correction dt_p on particles' position
            for (const int id: xpbd->ids) {
                _particles[id]->position += _particles[id]->force;
                _particles[id]->force *= 0;
            }
        }
    }
}

void PBD_System::step_constraint_jacobi(const scalar dt) {
    std::vector<int> counts(_particles.size(), 0);

    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        if (!xpbd->active()) continue;
        xpbd->apply(_particles, dt); // if xpbd

        for (const int id: xpbd->ids) {
            counts[id]++;
        }
    }

    for (int i = 0; i < _particles.size(); ++i) {
        Particle *&part = _particles[i];
        part->position += part->force / static_cast<scalar>(counts[i]);
        part->force *= 0;
    }
}
