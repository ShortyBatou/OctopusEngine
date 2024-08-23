#pragma once
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/Base/Solver.h"
#include <algorithm>
#include <random>
#include <iterator>

enum PBDSolverType {
    GaussSeidel, Jacobi, GaussSeidel_RNG
};

struct PBD_System : public ParticleSystem {
    PBD_System(Solver *solver, int nb_step, int nb_substep = 1, PBDSolverType solver_type = GaussSeidel,
               scalar global_damping = scalar(0)) : ParticleSystem(solver), _nb_step(nb_step), _nb_substep(nb_substep),
                                                    _type(solver_type), _global_damping(global_damping) {
        _groups.emplace_back();
    }

    void step(scalar dt) override;

    scalar get_residual(scalar dt) const;

    ~PBD_System() override;

    void clear_xpbd_constraints();

    int add_xpbd_constraint(XPBD_Constraint *constraint);

    void new_group();

    void draw_debug_xpbd() const;

    virtual void reset_lambda();

    virtual void update_velocity(scalar dt);

    virtual void step_constraint_gauss_seidel(scalar dt);

    virtual void step_constraint_gauss_seidel_rng(scalar dt);

    virtual void step_constraint_jacobi(scalar dt);

    bool _init = false;
    scalar _global_damping;
    int _nb_step, _nb_substep;
    PBDSolverType _type;

protected:
    std::vector<std::vector<XPBD_Constraint *> > _groups;
    std::vector<XPBD_Constraint *> _xpbd_constraints;
};
