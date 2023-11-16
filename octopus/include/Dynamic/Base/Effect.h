#pragma once

#include "Core/Base.h"
#include "Dynamic/Base/Effect.h"
#include <vector>

// Effect applied on all particles
struct Effect {
	Effect(scalar stiffness = 1., bool active = true) : _stiffness(stiffness), _active(active) {};
	virtual void init(const std::vector<Particle*>& particles) { }
	virtual void apply(const std::vector<Particle*>& particles, const scalar dt) = 0;

	bool active() { return _active; }
	void set_active(bool active) { _active = active; }

	scalar stiffness() { return _stiffness; }
	void set_stiffness(scalar stiffness) { _stiffness = stiffness; }
	virtual ~Effect() {}
protected:
	bool _active;
	scalar _stiffness;
};



struct ForceField : public Effect {
	enum Mode{Uniform, Linear, Quadratic };
	Vector3 position;
	scalar intencity;
	scalar radius;
	Mode mode;
	ForceField(Vector3 _position = Unit3D::Zero(), scalar _radius = 1., scalar _intencity = 1., Mode _mode = Linear) : position(_position), radius(_radius), intencity(_intencity), mode(_mode){ }

	virtual void apply(const std::vector<Particle*>& particles, const scalar dt) {
		for (Particle* p : particles) {
			Vector3 dir = p->position - position;
			scalar dist = glm::length(dir);
			if (dist <= radius) {
				switch (mode)
				{
					case Uniform: p->force += dir * (scalar(1.) / dist) * intencity; break;
					case Linear: p->force += dir * (scalar(1.) / radius) * intencity; break;
					case Quadratic: p->force += dir * intencity * (dist / (radius * radius)); break;
				}
			}
		}
	}
};

struct PlaneCollider : public Effect {
	Vector3 _o;
	Vector3 _n;
	PlaneCollider(const Vector3& o, const Vector3& n = Unit3D::up() , scalar stiffness = scalar(1.), bool active = true)
		: Effect(stiffness, active), _o(o), _n(n) 
	{}

	virtual void apply(const std::vector<Particle*>& particles, const scalar) override {
		Debug::SetColor(ColorBase::Blue());
		for (Particle* part : particles) {
			Vector3 op = part->position - _o;
			scalar d = glm::dot(op, _n);
			if (d > 0.) continue;

			part->position -= _n * d + _n * scalar(0.001);
		}
	}
};
