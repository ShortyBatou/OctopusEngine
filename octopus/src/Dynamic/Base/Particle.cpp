#pragma once
#include "Dynamic/Base/Particle.h"

void Particle::build(const Vector3 &_position, scalar _mass) {
    init_position = _position;
    position = _position;
    last_position = _position;
    offset = Unit3D::Zero();
    velocity = Unit3D::Zero();
    last_velocity = Unit3D::Zero();
    force = Unit3D::Zero();
    external_forces = Unit3D::Zero();
    mass = _mass;
    if (mass > 1e-8) {
        inv_mass = scalar(1.) / _mass;
    } else {
        inv_mass = scalar(0.);
    }
}

void Particle::reset() {
    Vector3 init_pos = init_position;
    build(init_pos, mass);
}
