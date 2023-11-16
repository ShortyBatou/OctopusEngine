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
    ParticleSystem(Solver* solver) : _solver(solver) { }

    virtual void init() {
        for (Effect* effect : _effects)
            effect->init(_particles);

        for (Constraint* constraint : _constraints) 
            constraint->init(_particles);        
    }

    virtual void step(const scalar dt) {
        step_solver(dt);
        step_effects(dt);
        step_constraint(dt);
        reset_external_forces();
    }

    virtual void reset_external_forces() {
        for (Particle* particle : _particles) {
            particle->external_forces *= scalar(0.);
        }
    }

    virtual void step_solver(const scalar dt) {
        for (Particle* particle : _particles) {
            if (!particle->active) continue;
            particle->last_position = particle->position;

            _solver->integrate(particle, dt);
        }
    }

    virtual void step_constraint(const scalar dt) {
        for (Constraint* constraint : _constraints) {
            if (!constraint->active()) continue;
            constraint->apply(_particles, dt);
        }
    }

    virtual void step_effects(const scalar dt) {
        for (Effect* effect : _effects) {
            if (!effect->active()) continue;
            effect->apply(_particles, dt);
        }
    }

    unsigned int add_particle(const Vector3& _position, scalar _mass) {
        _particles.push_back(new Particle(_position, _mass));
        return _particles.size() - 1;
    }

    unsigned int add_constraint(Constraint* constraint) {
        _constraints.push_back(constraint);
        return _constraints.size() - 1;
    }

    unsigned int add_effect(Effect* effect) {
        _effects.push_back(effect);
        return _constraints.size() - 1;
    }

    virtual void clear() {
        clear_particles();
        clear_effects();
        clear_constraints();
    }

    virtual void clear_particles() {
        for (Particle* p : _particles)
            delete p;
        _particles.clear();
    }

    virtual void clear_effects() {
        for (Effect* effect : _effects)
            delete effect;
        _effects.clear();
    }

    virtual void clear_constraints() {
        for (Constraint* c : _constraints)
            delete c;
        _constraints.clear();
    }

    virtual void reset() {
        for (Particle* p : _particles)
            p->reset();
    };

    virtual Particle* get(unsigned int i) {
        return _particles[i];
    }

protected:
    Solver* _solver;
    std::vector<Particle*> _particles;
    std::vector<Effect*> _effects;
    std::vector<Constraint*> _constraints;
};