#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/Base/Constraint.h"
#include <vector>


class XPBD_Constraint : public Constraint {
public:
    XPBD_Constraint(std::vector<unsigned int> ids, scalar stiffness, bool active = true) : Constraint(ids, stiffness, active), _lambda(0) {}
    virtual void init(const std::vector<Particle*>& particles) override {}
    virtual void apply(const std::vector<Particle*>& particles, const scalar dt) override {
        if (_stiffness <= 0) return;

        scalar w_sum(0.);
        std::vector<Particle*> x(this->nb());
        for (unsigned int i = 0; i < this->nb(); ++i) {
            x[i] = particles[this->_ids[i]];
            w_sum += x[i]->inv_mass;
        }

        scalar C = 0;
        std::vector<Vector3> grads(this->nb(), Unit3D::Zero());
        if (!project(x, grads, C)) return;

        scalar sum_norm_grad = 0;
        for (unsigned int i = 0; i < this->nb(); ++i) {
            sum_norm_grad += glm::dot(grads[i], grads[i]) * x[i]->inv_mass;
        }

        if (sum_norm_grad < 1e-24) return;
        scalar alpha = scalar(1.0) / (_stiffness * dt * dt);
        scalar dt_lambda = -(C + alpha * _lambda) / (sum_norm_grad + alpha);
        
        if (std::abs(dt_lambda) < 1e-24) return;
        _lambda += dt_lambda;

        for (unsigned int i = 0; i < this->nb(); ++i) {
            x[i]->add_force(dt_lambda * x[i]->inv_mass * grads[i]); // we use force to store dt_x
        }
    }

    virtual bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) { return false; }

    virtual ~XPBD_Constraint() {}

    void set_lambda(scalar lambda) { _lambda = lambda; }
protected:
    scalar _lambda;
};

class XPBD_DistanceConstraint : public XPBD_Constraint {
public:
    XPBD_DistanceConstraint(unsigned int a, unsigned int b, scalar stiffness, bool active = true) : XPBD_Constraint({a,b}, stiffness, active) {}

    virtual void init(const std::vector<Particle*>& particles) override {
        Vector3 pa = particles[this->_ids[0]]->position;
        Vector3 pb = particles[this->_ids[1]]->position;
        _rest_length = glm::distance(pa, pb);
    }

    virtual bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Vector3 pa = x[0]->position;
        Vector3 pb = x[1]->position;
        Vector3 n = pa - pb;
        scalar d = glm::l2Norm(n);
        if (d < 1e-6) return false;

        n /= d;
        C = d - _rest_length;

        if (std::abs(C) < 1e-6) return false;
        
        grads[0] = n;
        grads[1] = -n;

        return true;
    }
protected:
    scalar _rest_length;
};