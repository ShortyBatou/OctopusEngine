#pragma once
#include "Dynamic/Base/ParticleSystem.h"
#include "Manager/Debug.h"

void ParticleSystem::step(const scalar dt) {
    step_solver(dt);
    step_effects(dt);
    step_constraint(dt);
    reset_external_forces();
}

void ParticleSystem::reset_external_forces() {
    for (Particle *particle: _particles) {
        particle->external_forces *= scalar(0.);
    }
}

void ParticleSystem::step_solver(const scalar dt) {
    for (Particle *particle: _particles) {
        if (!particle->active) continue;
        particle->last_position = particle->position;

        _solver->integrate(particle, dt);
    }
}

void ParticleSystem::step_constraint(const scalar dt) {
    for (Constraint *constraint: _constraints) {
        if (!constraint->active()) continue;
        constraint->apply(_particles, dt);
    }
}

void ParticleSystem::step_effects(const scalar dt) {
    for (Effect *effect: _effects) {
        if (!effect->active()) continue;
        effect->apply(_particles, dt);
    }
}

int ParticleSystem::add_particle(const Vector3 &_position, scalar _mass) {
    _particles.push_back(new Particle(_position, _mass));
    return static_cast<int>(_particles.size()) - 1;
}

int ParticleSystem::add_constraint(Constraint *constraint) {
    _constraints.push_back(constraint);
    _constraints.back()->init(this->_particles);
    return static_cast<int>(_constraints.size()) - 1;
}

int ParticleSystem::add_effect(Effect *effect) {
    _effects.push_back(effect);
    _effects.back()->init(this->_particles);
    return static_cast<int>(_constraints.size()) - 1;
}

void ParticleSystem::clear_particles() {
    for (Particle *p: _particles)
        delete p;
    _particles.clear();
}

void ParticleSystem::clear_effects() {
    for (Effect *effect: _effects)
        delete effect;
    _effects.clear();
}

void ParticleSystem::clear_constraints() {
    for (Constraint *c: _constraints)
        delete c;
    _constraints.clear();
}

void ParticleSystem::reset() {
    for (Particle *p: _particles)
        p->reset();
};

void ParticleSystem::draw_debug_constraints() {
    for (Constraint *c: _constraints)
        c->draw_debug(_particles);
}

void ParticleSystem::draw_debug_effects() {
    for (Effect *e: _effects)
        e->draw_debug(_particles);
}

void ParticleSystem::draw_debug_particles(const Color &c) {
    Debug::SetColor(c);
    for (Particle *p: _particles) {
        Debug::Cube(p->position, 0.01f);
    }
}

ParticleSystem::~ParticleSystem() {
    clear_particles();
    clear_effects();
    clear_constraints();
    delete _solver;
}
