#pragma once
#include "Core/Base.h"
#include <mutex>

struct Particle {
    Particle(const Vector3 &_position, scalar _mass)
        : active(true), mass(0), inv_mass(0),
          init_position(), position(), last_position(), offset(),
          velocity(), last_velocity(),
          force(), external_forces(), mask(1) {
        build(_position, _mass);
    }

    std::mutex mutex;
    bool active;
    scalar mass, inv_mass;
    Vector3 init_position, position, last_position;
    Vector3 offset;
    Vector3 velocity, last_velocity;
    Vector3 force;
    Vector3 external_forces;
    int mask;

    void build(const Vector3 &_position, scalar _mass);
    void reset();
};
