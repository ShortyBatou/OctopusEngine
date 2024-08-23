#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Manager/Dynamic.h"

class Solver {
public:
    virtual ~Solver() = default;

    explicit Solver(const scalar damping = 1.) : _damping(damping) {
    }

    virtual void integrate(Particle *p, scalar dt) = 0;

    scalar &damping() { return _damping; }

protected:
    scalar _damping;
};

class EulerExplicit : public Solver {
public:
    explicit EulerExplicit(const scalar damping = 1.) : Solver(damping) {
    }

    void integrate(Particle *p, scalar dt) override;
};

class EulerSemiExplicit : public Solver {
public:
    explicit EulerSemiExplicit(const scalar damping = 1.) : Solver(damping) {
    }

    void integrate(Particle *p, scalar dt) override;
};

class AdaptiveEulerSemiExplicit : public Solver {
public:
    explicit AdaptiveEulerSemiExplicit(const scalar damping = 1.) : Solver(damping) {
    }

    void integrate(Particle *p, scalar dt) override;
};
