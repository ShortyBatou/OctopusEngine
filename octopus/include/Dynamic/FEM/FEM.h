#pragma once
#include "Dynamic/Base/Solver.h"
#include "Dynamic/Base/ParticleSystem.h"

class FEM_System : public ParticleSystem {
public:
    explicit FEM_System(Solver *solver, unsigned int nb_substep = 1) : ParticleSystem(solver), _nb_substep(nb_substep) {
    }

    void step(scalar dt) override;

    virtual void step_force(scalar dt);

    int add_fem(Constraint *constraint);

    void clear_fem();

    ~FEM_System() override;

protected:
    std::vector<Constraint *> _fems;
    unsigned int _nb_substep;
};
