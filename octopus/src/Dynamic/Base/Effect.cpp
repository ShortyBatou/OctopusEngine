#pragma once
#include "Dynamic/Base/Effect.h"
#include "Manager/Debug.h"

void ForceField::apply(const std::vector<Particle *> &particles, const scalar dt) {
    for (Particle *p: particles) {
        Vector3 dir = p->position - position;
        scalar dist = glm::length(dir);
        if (dist <= radius) {
            switch (mode) {
                case Uniform: p->force += dir * (1.f / dist) * intensity;
                    break;
                case Linear: p->force += dir * (1.f / radius) * intensity;
                    break;
                case Quadratic: p->force += dir * intensity * (dist / (radius * radius));
                    break;
            }
        }
    }
}


void PlaneConstraint::apply(const std::vector<Particle *> &particles, const scalar) {
    Debug::SetColor(ColorBase::Blue());
    for (Particle *part: particles) {
        Vector3 op = part->position - _o;
        const scalar d = glm::dot(op, _n);
        if (d > 0.) continue;
        part->position -= _n * d - _n * 0.001f;
    }
}
