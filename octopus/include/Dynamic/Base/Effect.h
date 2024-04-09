#pragma once

#include "Core/Base.h"
#include "Dynamic/Base/Effect.h"
#include "Manager/Debug.h"
#include <vector>

// Effect applied on all particles
struct Effect {
	Effect(scalar stiffness = 1., bool active = true) : _stiffness(stiffness), _active(active) {};
	virtual void init(const std::vector<Particle*>& particles) { }
	virtual void apply(const std::vector<Particle*>& particles, const scalar dt) = 0;
	virtual void draw_debug(const std::vector<Particle*>& parts) { }

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

struct PlaneConstraint : public Effect {
	Vector3 _o;
	Vector3 _n;
	PlaneConstraint(const Vector3& o, const Vector3& n = Unit3D::up() , scalar stiffness = scalar(1.), bool active = true) : Effect(stiffness, active), _o(o), _n(n) 
	{}

	virtual void apply(const std::vector<Particle*>& particles, const scalar) override {
		Debug::SetColor(ColorBase::Blue());
		for (Particle* part : particles) {
			Vector3 op = part->position - _o;
			scalar d = glm::dot(op, _n);
			if (d > 0.) continue;

			part->position -= _n * d - _n * scalar(0.001);
		}
	}
};


struct Sin_ForceField : public Effect {
	scalar lambda, mu, rho;
	scalar global_error;
	scalar scale;
	Sin_ForceField(scalar _lambda, scalar _mu, scalar _rho, bool active = true) : Effect(1.f, active), lambda(_lambda), mu(_mu), rho(_rho), scale(0.2f)
	{}

	virtual void apply(const std::vector<Particle*>& particles, const scalar dt) override {
		scalar pi = 3.14159265358979323846f;
		Vector3 f;
		scalar x, y, z;
		global_error = 0;
		int nb = 0;
		scalar t = Time::Fixed_Timer();
		for (Particle* part : particles) {
			x = part->init_position.x;
			y = part->init_position.y;
			z = part->init_position.z;

			f.x = 4.f * pi * pi * (-(lambda + mu) * (sin(2.f * pi * y) * cos(2.f * pi * z) + sin(2.f * pi * z) * cos(2.f * pi * y)) * cos(2.f * pi * x)
				+ 4.f * sin(2.f * pi * x) * sin(2 * pi * y) * sin(2 * pi * z) * (1.f / 4.f * lambda + mu));

			f.y = 4.f * pi * pi * (-(lambda + mu) * (sin(2.f * pi * z) * cos(2.f * pi * x) + sin(2.f * pi * x) * cos(2.f * pi * z)) * cos(2.f * pi * y)
				+ 4.f * sin(2.f * pi * x) * sin(2.f * pi * y) * sin(2.f * pi * z) * (1.f / 4.f * lambda + mu));

			f.z = 4.f * pi * pi * (-(lambda + mu) * (sin(2.f * pi * x) * cos(2.f * pi * y) + sin(2.f * pi * y) * cos(2.f * pi * x)) * cos(2.f * pi * z)
				+ 4.f * sin(2.f * pi * x) * sin(2.f * pi * y) * sin(2.f * pi * z) * (1.f / 4.f * lambda + mu));

			part->external_forces = f * scale * std::min(1.f, t) / rho;
		}
	}

	Vector3 get_analytical(scalar x, scalar y, scalar z) {
		scalar pi = 3.14159265358979323846;
		return Vector3(
			sin(2.f * pi * x) * sin(2.f * pi * y) * sin(2.f * pi * z),
			sin(2.f * pi * x) * sin(2.f * pi * y) * sin(2.f * pi * z),
			sin(2.f * pi * x) * sin(2.f * pi * y) * sin(2.f * pi * z)) * scale;
	}

};
