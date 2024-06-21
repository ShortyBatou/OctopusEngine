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

class AdaptativeEulerSemiExplicit : public Solver {
public:
	AdaptativeEulerSemiExplicit(scalar damping = 1.) : Solver(damping) { }
	void integrate(Particle* p, const scalar dt) override {
		if (p->mass <= eps) return;
		Vector3 a = (p->velocity - p->last_velocity) / dt;
		Vector3 a_ext = ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity());
		scalar n_a_ext = glm::length(a_ext);
		scalar s = 1.;
		if (n_a_ext > eps) {
			Vector3 a_hat = a_ext / n_a_ext;
			s = glm::dot(a, a_hat);
			s = (s > n_a_ext) ? 1.f : s;
			s = (s < 0) ? 0.f : s;
		}
		p->last_velocity = p->velocity;
		p->velocity += s * a_ext * dt;
		p->velocity *= _damping;
		p->position += p->velocity * dt;
		p->force *= 0.;
	}
};