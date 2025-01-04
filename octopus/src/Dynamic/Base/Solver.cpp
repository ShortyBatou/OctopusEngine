#pragma once
#include "Dynamic/Base/Solver.h"

void EulerExplicit::integrate(Particle *p, const scalar dt) {
    if (p->mass <= eps) return;
    p->position += p->velocity * dt;
    p->velocity += ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt;
    p->velocity *= _damping;
    p->force *= 0.;
}

void EulerSemiExplicit::integrate(Particle *p, const scalar dt) {
    if (p->mass <= eps) return;
    p->velocity += ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt;
    p->velocity *= _damping;
    p->position += p->velocity * dt;
    p->force *= 0.;
}


void AdaptiveEulerSemiExplicit::integrate(Particle *p, const scalar dt) {
    if (p->mass <= eps) return;
    const Vector3 a = (p->velocity - p->last_velocity) / dt;
    const Vector3 a_ext = ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity());
    scalar n_a_ext = glm::length(a_ext);
    scalar s = 1.;
    if (n_a_ext > eps) {
        Vector3 a_hat = a_ext / n_a_ext;
        s = glm::dot(a, a_hat);
        s = (s > n_a_ext) ? 1.f : s;
        s = (s < 0) ? 0.f : s;
    }
    p->last_velocity = p->velocity;
    p->velocity += s * a_ext * dt;
    p->velocity *= _damping;
    p->position += p->velocity * dt;
    p->force *= 0.;
}
