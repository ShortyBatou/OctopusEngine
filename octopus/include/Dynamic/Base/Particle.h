#pragma once
#include "Core/Base.h"
#include <mutex>
struct Particle {
    Particle(const Vector3& _position, scalar _mass) : active(true) {
        build(_position, _mass);
	}

    std::mutex mutex;
    bool active;
	scalar mass, inv_mass;
    Vector3 init_position, position, last_position;
	Vector3 offset;
	Vector3 velocity;
	Vector3 force;
    Vector3 external_forces;

    void add_force(const Vector3& f) {
        mutex.lock(); force += f; mutex.unlock();
    }

    void build(const Vector3& _position, scalar _mass) {
        init_position = _position;
        position = _position;
        last_position = _position;
        offset = Unit3D::Zero();
        velocity = Unit3D::Zero();
        force = Unit3D::Zero();
        external_forces = Unit3D::Zero();
        mass = _mass;
        if (mass > 1e-8) {
            inv_mass = scalar(1.) / _mass;
        }
        else {
            inv_mass = scalar(0.);
        }
    }

    void reset() {
        Vector3 init_pos = init_position;
        build(init_pos, mass);
    }
};