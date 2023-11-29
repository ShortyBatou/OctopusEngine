#pragma once
#include "Dynamic/Base/Solver.h"
#include "Dynamic/Base/ParticleSystem.h"

class FEM_System : public ParticleSystem {
public:
    FEM_System(Solver* solver, unsigned int nb_substep = 1) :
        ParticleSystem(solver), _nb_substep(nb_substep)
    { }

    virtual void step(const scalar dt) override {
        scalar h = dt / (scalar)_nb_substep;
        this->step_effects(dt);
        for (unsigned int i = 0; i < _nb_substep; i++)
        {
            this->step_force(h);
            this->step_solver(h);
            this->step_constraint(dt);
        }
    }

    virtual ~FEM_System() {
        clear_fem();
    }

    void clear_fem() {
        for (Constraint* c : _fems) delete c;
        _fems.clear();
    }

    unsigned int add_fem(Constraint* constraint) {
        _fems.push_back(constraint);
        _fems.back()->init(this->_particles);
        return _fems.size();
    }

protected:

    virtual void step_force(const scalar dt) {
        for (Constraint* fem : _fems) {
            if (!fem->active()) continue;
            fem->apply(this->_particles, dt); // if xpbd
        }
    }

protected:
    std::vector<Constraint*> _fems;
    unsigned int _nb_substep;
};