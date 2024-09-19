#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/Base/Effect.h"
#include "Dynamic/Base/Constraint.h"
#include <vector>
#include "Solver.h"

class ParticleSystem {
public:
    explicit ParticleSystem(Solver *solver) : _solver(solver) {
    }

    virtual void step(scalar dt);

    virtual void step_solver(scalar dt);

    virtual void step_constraint(scalar dt);

    virtual void step_effects(scalar dt);

    virtual void reset_external_forces();

    int add_particle(const Vector3 &_position, scalar _mass);

    int add_constraint(Constraint *constraint);

    int add_effect(Effect *effect);

    void clear_particles();

    void clear_effects();

    void clear_constraints();

    void reset();

    virtual void draw_debug_constraints();

    virtual void draw_debug_effects();

    virtual void draw_debug_particles(const Color &c);

    [[nodiscard]] Particle *get(int i) { return _particles[i]; }
    [[nodiscard]] std::vector<Particle *> &particles() { return _particles; }
    [[nodiscard]] int nb_particles() const { return static_cast<int>(_particles.size()); }
    [[nodiscard]] Solver *solver() const { return _solver; }

    virtual ~ParticleSystem();

protected:
    Solver *_solver;
    std::vector<Particle *> _particles;
    std::vector<Effect *> _effects;
    std::vector<Constraint *> _constraints;
};
