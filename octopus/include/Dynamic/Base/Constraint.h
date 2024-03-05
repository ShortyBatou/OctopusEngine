#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Effect.h"
#include <vector>

/// Effect applied on some particles
struct Constraint : public Effect {
    Constraint(std::vector<unsigned int> ids, scalar stiffness = 1., bool active = true) : Effect(stiffness, active), _ids(ids) {}
    virtual void init(const std::vector<Particle*>& particles) override { };
    virtual void apply(const std::vector<Particle*>& particles, const scalar dt) override = 0;
    unsigned int nb() { return _ids.size(); }
    const std::vector<unsigned int>& ids() { return _ids; }
    virtual ~Constraint() {}
protected:
    std::vector<unsigned int> _ids; // particles that are conserned by this PBD_Constraint
};

struct FixPoint : public Constraint {
    FixPoint(unsigned int id, scalar stiffness = scalar(1.), bool active = true) : Constraint({ id }, stiffness, active) {}

    virtual void apply(const std::vector<Particle*>& parts, const scalar) override {
        parts[this->_ids[0]]->reset();
    }
};

struct RB_Fixation : public Constraint {
    Matrix3x3 rot;
    Vector3 com, offset;
    RB_Fixation(std::vector<unsigned int> ids, scalar stiffness = scalar(1.), bool active = true) : Constraint(ids, stiffness, active), rot(Matrix::Identity3x3()), offset(Unit3D::Zero()){}

    virtual void init(const std::vector<Particle*>& parts) override {
        Vector3 sum_position(0.0f, 0.0f, 0.0f);
        for (unsigned int i = 0; i < this->nb(); i++) {
            Particle* part = parts[this->_ids[i]];
            sum_position += part->position;
        }
        com = sum_position / scalar(this->_ids.size());
    }

    virtual void apply(const std::vector<Particle*>& parts, const scalar) override {
        for (unsigned int i = 0; i < this->nb(); i++) {
            Particle* part = parts[this->_ids[i]];
            Vector3 target = offset + com + rot * (part->init_position - com);
            part->position += (target - part->position) * this->_stiffness;
            part->velocity *= 0;
            part->force *= 0;
        }
    }
    virtual void draw_debug(const std::vector<Particle*>& parts) {
        Debug::Axis(com, rot, 0.1);
        Debug::SetColor(ColorBase::Blue());
        for (unsigned int i = 0; i < this->nb(); i++) {
            Particle* part = parts[this->_ids[i]];
            Debug::Cube(parts[this->_ids[i]]->position, 0.02f);
        }
    }

};


struct ConstantForce : public Constraint {
    Vector3 f;
    ConstantForce(std::vector<unsigned int> ids, Vector3 force, bool active = true) : Constraint(ids, 1., active), f(force) {}

    virtual void init(const std::vector<Particle*>& parts) override {  }

    virtual void apply(const std::vector<Particle*>& parts, const scalar) override {
        for (unsigned int i = 0; i < this->nb(); i++) {
            Particle* part = parts[this->_ids[i]];
            part->force += f;
        }
    }
    virtual void draw_debug(const std::vector<Particle*>& parts) {
        Debug::SetColor(ColorBase::Red());
        for (unsigned int i = 0; i < this->nb(); i++) {
            Particle* part = parts[this->_ids[i]];
            Debug::Line(part->position, part->position + f * scalar(0.1));
        }
    }

};


