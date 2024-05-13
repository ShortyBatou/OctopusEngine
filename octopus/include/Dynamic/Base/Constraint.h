#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Effect.h"
#include <vector>

/// Effect applied on some particles
struct Constraint : public Effect {
    Constraint(std::vector<int> _ids, scalar _stiffness = 1., bool _active = true) : Effect(_stiffness, _active), ids(_ids) {}
    virtual void init(const std::vector<Particle*>& particles) override { };
    virtual void apply(const std::vector<Particle*>& particles, const scalar dt) override = 0;
    int nb() { return ids.size(); }

    std::vector<Particle*> get_particles(const std::vector<Particle*>& particles) {
        std::vector<Particle*> p(nb());
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] = particles[ids[i]];
        }
        return p;
    }

    virtual ~Constraint() {}
    std::vector<int> ids; // particles that are conserned by this PBD_Constraint
};

struct FixPoint : public Constraint {
    FixPoint(int id, scalar stiffness = scalar(1.), bool active = true) : Constraint({ id }, stiffness, active) {}

    virtual void apply(const std::vector<Particle*>& parts, const scalar) override {
        parts[ids[0]]->reset();
    }
};

struct RB_Fixation : public Constraint {
    Matrix3x3 rot;
    Vector3 com, offset;
    RB_Fixation(std::vector<int> ids, scalar stiffness = scalar(1.), bool active = true) : Constraint(ids, stiffness, active), rot(Matrix::Identity3x3()), offset(Unit3D::Zero()){}

    virtual void init(const std::vector<Particle*>& parts) override {
        Vector3 sum_position(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < this->nb(); i++) {
            Particle* part = parts[ids[i]];
            sum_position += part->position;
        }
        com = sum_position / scalar(nb());
    }

    virtual void apply(const std::vector<Particle*>& parts, const scalar) override {
        for (int i = 0; i < this->nb(); i++) {
            Particle* part = parts[ids[i]];
            Vector3 target = offset + com + rot * (part->init_position - com);
            part->position += (target - part->position) * this->_stiffness;
            part->velocity *= 0.f;
            part->force *= 0.f;
        }
    }
    virtual void draw_debug(const std::vector<Particle*>& parts) {
        Debug::Axis(com, rot, 0.1f);
        Debug::SetColor(ColorBase::Blue());
        for (int i = 0; i < this->nb(); i++) {
            Particle* part = parts[ids[i]];
            Debug::Cube(parts[ids[i]]->position, 0.02f);
        }
    }

};


struct ConstantForce : public Constraint {
    Vector3 f;
    ConstantForce(std::vector<int> ids, Vector3 force, bool active = true) : Constraint(ids, 1., active), f(force) {}

    virtual void init(const std::vector<Particle*>& parts) override {  }

    virtual void apply(const std::vector<Particle*>& parts, const scalar) override {
        for (int i = 0; i < this->nb(); i++) {
            Particle* part = parts[ids[i]];
            part->force += f;
        }
    }
    virtual void draw_debug(const std::vector<Particle*>& parts) {
        Debug::SetColor(ColorBase::Red());
        for (int i = 0; i < this->nb(); i++) {
            Particle* part = parts[ids[i]];
            Debug::Line(part->position, part->position + f * scalar(0.1));
        }
    }

};


