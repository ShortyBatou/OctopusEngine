#pragma once

#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include <vector>

// Effect applied on all particles
struct Effect {
    Effect(scalar stiffness = 1., bool active = true) : _stiffness(stiffness), _active(active) {
    };

    virtual void init(const std::vector<Particle *> &particles) {
    }

    virtual void apply(const std::vector<Particle *> &particles, scalar dt) = 0;

    virtual void draw_debug(const std::vector<Particle *> &parts) {
    }

    bool active() const { return _active; }
    void set_active(bool active) { _active = active; }

    scalar stiffness() const { return _stiffness; }
    void set_stiffness(scalar stiffness) { _stiffness = stiffness; }

    virtual ~Effect() = default;

protected:
    bool _active;
    scalar _stiffness;
};


struct ForceField : public Effect {
    enum Mode { Uniform, Linear, Quadratic };

    Vector3 position;
    scalar intensity;
    scalar radius;
    Mode mode;

    explicit ForceField(Vector3 _position = Unit3D::Zero(), scalar _radius = 1., scalar _intencity = 1.,
                        Mode _mode = Linear)
        : position(_position), radius(_radius), intensity(_intencity), mode(_mode) {
    }

    virtual void apply(const std::vector<Particle *> &particles, const scalar dt);
};

struct PlaneConstraint : public Effect {
    Vector3 _o;
    Vector3 _n;

    explicit PlaneConstraint(const Vector3 &o, const Vector3 &n = Unit3D::up(), scalar stiffness = scalar(1.),
                             bool active = true) : Effect(stiffness, active), _o(o), _n(n) {
    }

    void apply(const std::vector<Particle *> &particles, scalar) override;
};
