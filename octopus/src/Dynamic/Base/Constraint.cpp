#pragma once
#include "Dynamic/Base/Constraint.h"
#include <vector>
#include <Manager/Debug.h>

std::vector<Particle *> Constraint::get_particles(const std::vector<Particle *> &particles) const {
    std::vector<Particle *> p(nb());
    for (size_t i = 0; i < p.size(); ++i) {
        p[i] = particles[ids[i]];
    }
    return p;
}


void FixPoint::apply(const std::vector<Particle *> &parts, const scalar) {
    parts[ids[0]]->reset();
}


void RB_Fixation::init(const std::vector<Particle *> &parts) {
    Vector3 sum_position(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < this->nb(); i++) {
        Particle *part = parts[ids[i]];
        part->active = false;
        sum_position += part->position;
    }
    com = sum_position / static_cast<scalar>(nb());
}

void RB_Fixation::apply(const std::vector<Particle *> &parts, const scalar) {
    for (int i = 0; i < this->nb(); i++) {
        Particle *part = parts[ids[i]];
        Vector3 target = offset + com + rot * (part->init_position - com);
        part->position += (target - part->position) * this->_stiffness;
        part->velocity *= 0.f;
        part->force *= 0.f;
    }
}

void RB_Fixation::draw_debug(const std::vector<Particle *> &parts) {
    Debug::Axis(com, rot, 0.1f);
    Debug::SetColor(ColorBase::Blue());
    for (int i = 0; i < this->nb(); i++) {
        Debug::Cube(parts[ids[i]]->position, 0.02f);
    }
}


void ConstantForce::apply(const std::vector<Particle *> &parts, const scalar) {
    for (int i = 0; i < this->nb(); i++) {
        Particle *part = parts[ids[i]];
        part->force += f;
    }
}

void ConstantForce::draw_debug(const std::vector<Particle *> &parts) {
    Debug::SetColor(ColorBase::Red());
    for (int i = 0; i < this->nb(); i++) {
        const Particle *part = parts[ids[i]];
        Debug::Line(part->position, part->position + f * 0.1f);
    }
}
