#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Manager/Dynamic.h"

class Solver {
public:
	Solver(scalar damping = 1.) : _damping(damping) { }
	virtual void integrate(Particle* p, const scalar dt) = 0;
	scalar& damping() { return _damping; }
protected:
	scalar _damping;
};

class EulerExplicit : public Solver {
public:
	EulerExplicit(scalar damping = 1.) : Solver(damping) { }
	void integrate(Particle* p, const scalar dt) override {
		if (p->mass <= eps) return;
		p->position += p->velocity * dt;
		p->velocity += ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt;
		p->velocity *= _damping;
		p->force *= 0.;

	}
};

class EulerSemiExplicit : public Solver {
public:
	EulerSemiExplicit(scalar damping = 1.) : Solver(damping) { }
	void integrate(Particle* p, const scalar dt) override {
		if (p->mass <= eps) return;
		p->velocity += ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt;
		p->velocity *= _damping;
		p->position += p->velocity * dt;
		p->force *= 0.;
	}
};