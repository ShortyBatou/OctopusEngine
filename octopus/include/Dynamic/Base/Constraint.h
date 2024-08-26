#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Effect.h"
#include "Dynamic/Base/Particle.h"
#include <vector>

/// Effect applied on some particles
struct Constraint : Effect {
    std::vector<int> ids; // particles that are conserned by this PBD_Constraint

    explicit Constraint(const std::vector<int>& _ids, scalar _stiffness = 1., bool _active = true) : Effect(_stiffness, _active), ids(_ids) {}
    void init(const std::vector<Particle*>& particles) override { };
    void apply(const std::vector<Particle*>& particles, scalar dt) override = 0;
    [[nodiscard]] int nb() const { return static_cast<int>(ids.size()); }
    [[nodiscard]] std::vector<Particle*> get_particles(const std::vector<Particle*>& particles) const;

    ~Constraint() override = default;
};

struct FixPoint final : Constraint {
    explicit FixPoint(int id, scalar stiffness = 1., bool active = true) : Constraint({ id }, stiffness, active) {}
    void apply(const std::vector<Particle*>& parts, scalar) override;
};

struct RB_Fixation final : Constraint {
    Matrix3x3 rot;
    Vector3 com, offset;

    explicit RB_Fixation(const std::vector<int>& ids, scalar stiffness = 1., bool active = true)
        : Constraint(ids, stiffness, active), rot(Matrix::Identity3x3()), com(Unit3D::Zero()), offset(Unit3D::Zero()){}

    void init(const std::vector<Particle*>& parts) override;
    void apply(const std::vector<Particle*>& parts, scalar) override;
    void draw_debug(const std::vector<Particle*>& parts) override;
};


struct ConstantForce final : Constraint {
    Vector3 f;
    explicit ConstantForce(const std::vector<int>& ids, Vector3 force, bool active = true) : Constraint(ids, 1., active), f(force) {}
    void apply(const std::vector<Particle*>& parts, scalar) override;
    void draw_debug(const std::vector<Particle*>& parts) override;
};


