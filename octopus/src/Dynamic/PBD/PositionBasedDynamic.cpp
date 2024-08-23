#include "Dynamic/PBD/PositionBasedDynamic.h"
#include <algorithm>
#include <random>
#include <iterator>


void PBD_System::step(const scalar dt) {
    scalar h = dt / static_cast<scalar>(_nb_substep);
    for (int i = 0; i < _nb_substep; i++) {
        this->step_solver(h);
        this->reset_lambda();
        for (int j = 0; j < _nb_step; ++j) {
            if (_type == Jacobi)
                step_constraint_jacobi(h);
            if (_type == GaussSeidel_RNG)
                step_constraint_gauss_seidel_rng(h);
            else
                step_constraint_gauss_seidel(h);
        }

        this->step_constraint(dt); // optional

        this->step_effects(dt); // optional

        this->update_velocity(h);
    }
    this->reset_external_forces();
}

scalar PBD_System::get_residual(const scalar dt) const {
    scalar sub_dt = dt / static_cast<scalar>(_nb_substep);
    scalar residual = 0;
    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        residual += xpbd->get_dual_residual(this->_particles, sub_dt);
    }
    return residual / static_cast<scalar>(_xpbd_constraints.size());
}

PBD_System::~PBD_System() {
    clear_xpbd_constraints();
}

void PBD_System::clear_xpbd_constraints() {
    for (XPBD_Constraint *c: _xpbd_constraints) delete c;
    _xpbd_constraints.clear();
}

int PBD_System::add_xpbd_constraint(XPBD_Constraint *constraint) {
    _xpbd_constraints.push_back(constraint);
    _xpbd_constraints.back()->init(this->_particles);
    _groups[_groups.size() - 1].push_back(constraint);
    return static_cast<int>(_xpbd_constraints.size());
}

void PBD_System::new_group() {
    _groups.emplace_back();
}

void PBD_System::draw_debug_xpbd() const {
    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        xpbd->draw_debug(this->_particles);
    }
}


void PBD_System::reset_lambda() {
    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        xpbd->set_lambda(0);
    }
}

void PBD_System::update_velocity(const scalar dt) {
    for (Particle *p: this->_particles) {
        if (!p->active) continue;
        p->velocity = (p->position - p->last_position) / dt;

        scalar norm_v = glm::length(p->velocity);
        if (norm_v > 1e-12) {
            scalar damping = -norm_v * std::min(scalar(1), _global_damping * dt * p->inv_mass);
            p->velocity += glm::normalize(p->velocity) * damping;
        }
    }
}

void PBD_System::step_constraint_gauss_seidel(const scalar dt) {
    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        if (!xpbd->active()) continue;

        // compute corrections
        xpbd->apply(this->_particles, dt);

        // apply correction dt_p on particles' position
        for (int id: xpbd->ids) {
            this->_particles[id]->position += this->_particles[id]->force; // here: force = delta x
            this->_particles[id]->force *= 0;
        }
    }
}

void PBD_System::step_constraint_gauss_seidel_rng(const scalar dt) {
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(_groups), std::end(_groups), rng);

    for (std::vector<XPBD_Constraint *> &group: _groups) {
        std::reverse(group.begin(), group.end());
        for (XPBD_Constraint *xpbd: group) {
            if (!xpbd->active()) continue;

            // compute corrections
            xpbd->apply(this->_particles, dt);

            // apply correction dt_p on particles' position
            for (int id: xpbd->ids) {
                this->_particles[id]->position += this->_particles[id]->force;
                this->_particles[id]->force *= 0;
            }
        }
    }
}

void PBD_System::step_constraint_jacobi(const scalar dt) {
    std::vector<int> counts(this->_particles.size(), 0);

    for (XPBD_Constraint *xpbd: _xpbd_constraints) {
        if (!xpbd->active()) continue;
        xpbd->apply(this->_particles, dt); // if xpbd

        for (int id: xpbd->ids) {
            counts[id]++;
        }
    }

    for (int i = 0; i < this->_particles.size(); ++i) {
        Particle *&part = this->_particles[i];
        part->position += part->force / scalar(counts[i]);
        part->force *= 0;
    }
}
