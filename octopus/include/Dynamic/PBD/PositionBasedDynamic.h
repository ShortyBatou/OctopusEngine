#pragma once
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/Base/Solver.h"
#include <algorithm>
#include <random>
enum PBDSolverType {
    GaussSeidel, Jacobi
};

struct PBD_System : public ParticleSystem {
    PBD_System(Solver* solver, unsigned int nb_step, unsigned int nb_substep = 1, PBDSolverType solver_type = GaussSeidel) :
        ParticleSystem(solver), _nb_step(nb_step), _nb_substep(nb_substep), _type(solver_type)
    { }

    virtual void init() override {
        for (Effect* effect : this->_effects)
            effect->init(this->_particles);

        for (Constraint* constraint : this->_constraints)
            constraint->init(this->_particles);

        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            xpbd->init(this->_particles);
        }
    }

    virtual void step(const scalar dt) override {
        scalar h = dt / (scalar)_nb_substep;
        for (unsigned int i = 0; i < _nb_substep; i++)
        {
            this->step_solver(h);
            this->reset_lambda();
            for (unsigned int j = 0; j < _nb_step; ++j) {
                if (_type == Jacobi)
                    step_constraint_jacobi(h);
                else
                    step_constraint_gauss_seidel(h);
            }

            this->step_constraint(dt);

            this->step_effects(dt);

            this->update_velocity(h);
        }
        this->reset_external_forces();
    }

    virtual ~PBD_System() {
        clear_xpbd_constraints();
    }

    void clear_xpbd_constraints() {
        for (XPBD_Constraint* c : _xpbd_constraints) delete c;
        _xpbd_constraints.clear();
    }

    unsigned int add_xpbd_constraint(XPBD_Constraint* constraint) {
        _xpbd_constraints.push_back(constraint);
        return _xpbd_constraints.size();
    }

    void draw_debug_xpbd() {
        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            xpbd->draw_debug(this->_particles);
        }
    }

protected:
    virtual void reset_lambda() {
        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            xpbd->set_lambda(0);
        }
    }

    virtual void update_velocity(scalar dt) {
        for(Particle* p : this->_particles)
        {
            if (!p->active) continue;
            p->velocity = (p->position - p->last_position) / dt;
        }
    }

    virtual void step_constraint_gauss_seidel(const scalar dt) {
        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            if (!xpbd->active()) continue;
            xpbd->apply(this->_particles, dt); // if xpbd

            for (unsigned int id : xpbd->ids()) {
                this->_particles[id]->position += this->_particles[id]->force;
                this->_particles[id]->force *= 0;
            }
        }
    }

    virtual void step_constraint_jacobi(const scalar dt) {
        std::vector<unsigned int> counts(this->_particles.size(), 0);

        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            if (!xpbd->active()) continue;
            xpbd->apply(this->_particles, dt); // if xpbd

            for (unsigned int id : xpbd->ids()) {
                _particles[id]->mutex.lock();
                counts[id]++;
                _particles[id]->mutex.unlock();
            }
        }

        for (unsigned int i = 0; i < this->_particles.size(); ++i) 
        {
            Particle*& part = this->_particles[i];
            part->position += part->force / scalar(counts[i]);
            part->force *= 0;
        }
    }

protected:
    std::vector<XPBD_Constraint*> _xpbd_constraints;
    unsigned int _nb_step, _nb_substep;
    PBDSolverType _type;
};
