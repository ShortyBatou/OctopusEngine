#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
class Solver {
public:
	Solver(const Vector3& gravity = Vector3(0.,-9.81,0.), scalar damping = 1.) : _gravity(gravity), _damping(damping) { }
	virtual void integrate(Particle* p, const scalar dt) = 0;
	Vector3& gravity() { return _gravity; }
	scalar& damping() { return _damping; }
protected:
	Vector3 _gravity;
	scalar _damping;
};

class EulerExplicit : public Solver {
public:
	EulerExplicit(const Vector3& gravity = Vector3(0., -9.81, 0.), scalar damping = 1.) : Solver(gravity, damping) { }
	void integrate(Particle* p, const scalar dt) override {
		if (p->mass <= 0.0) return;
		p->position += p->velocity * dt;
		p->velocity += ((p->force + p->external_forces) * p->inv_mass + this->_gravity) * dt;
		p->velocity *= _damping;
		p->force *= 0.;

	}
};


class EulerSemiExplicit : public Solver {
public:
	EulerSemiExplicit(const Vector3& gravity = Vector3(0., -9.81, 0.), scalar damping = 1.) : Solver(gravity, damping) { }
	void integrate(Particle* p, const scalar dt) override {
		if (p->mass <= 1e-12) return;
		p->velocity += ((p->force + p->external_forces) * p->inv_mass + this->_gravity) * dt;
		p->velocity *= _damping;
		p->position += p->velocity * dt;
		p->force *= 0.;
	}
};