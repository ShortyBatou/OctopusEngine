#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/Base/Constraint.h"
#include <vector>


class XPBD_Constraint : public Constraint {
public:
    explicit XPBD_Constraint(const std::vector<int>& ids, scalar stiffness, const bool active = true) : Constraint(ids, stiffness, active), _lambda(0) {}
    void apply(const std::vector<Particle*>& particles, scalar dt) override;
    virtual scalar get_dual_residual(const std::vector<Particle*>& particles, scalar dt);
    virtual bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) { return false; }

    void set_lambda(scalar lambda) { _lambda = lambda; }
protected:
    scalar _lambda;
};

class XPBD_DistanceConstraint final : public XPBD_Constraint {
public:
    explicit XPBD_DistanceConstraint(int a, int b, scalar stiffness, bool active = true) : XPBD_Constraint({a,b}, stiffness, active) {}

    void init(const std::vector<Particle*>& particles) override;
    bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override;
protected:
    scalar _rest_length{};
};