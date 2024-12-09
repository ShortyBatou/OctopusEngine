#pragma once
#include <random>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include <Dynamic/VBD/VertexBlockDescent.h>
#include <Manager/Debug.h>
#include <Manager/Input.h>



void VertexBlockDescent::chebyshev_acceleration(const int it, scalar &omega) {
    if (static_cast<int>(prev_prev_x.size()) == 0) {
        prev_prev_x.resize(nb_particles());
        prev_x.resize(nb_particles());
    }

    omega = compute_omega(omega, it);
    for (int i = 0; i < nb_particles(); ++i) {
        Particle *p = get(i);
        if (!p->active) continue;
        if (it >= 2) p->position = omega * (p->position - prev_prev_x[i]) + prev_prev_x[i];
        prev_prev_x[i] = prev_x[i];
        prev_x[i] = p->position;
    }
}

void VertexBlockDescent::step(const scalar dt) {
    const scalar sub_dt = dt / static_cast<scalar>(_sub_iteration);
    for (int i = 0; i < _sub_iteration; ++i) {
        for(VBD_Object* obj : _objs) obj->compute_inertia(this, sub_dt);
        // get the first guess
        step_solver(sub_dt);
        scalar omega = 0;
        for (int j = 0; j < _iteration; ++j) {
            for(VBD_Object* obj : _objs) obj->solve(this, sub_dt);
            //chebyshev_acceleration(j, omega);
        }
        step_effects(sub_dt);
        step_constraint(sub_dt);
        update_velocity(sub_dt);
    }
    reset_external_forces();
}

void VertexBlockDescent::update_velocity(const scalar dt) const {
    for (Particle *p: this->_particles) {
        if (!p->active) continue;
        p->velocity = (p->position - p->last_position) / dt;

    }
}