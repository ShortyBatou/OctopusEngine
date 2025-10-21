#include "Dynamic/PBD/XPBD_Constraint.h"
#include <vector>

void XPBD_Constraint::apply(const std::vector<Particle*>& particles, const scalar dt) {
    if (_stiffness <= 0) return;

    std::vector<Particle*> x(nb());
    for (int i = 0; i < nb(); ++i) {
        x[i] = particles[ids[i]];
    }

    scalar C = 0.f;
    std::vector grads(nb(), Unit3D::Zero());
    if (!project(x, grads, C)) return;

    scalar sum_norm_grad = 0.f;
    for (int i = 0; i < nb(); ++i) {
        sum_norm_grad += glm::dot(grads[i], grads[i]) * x[i]->inv_mass;
    }

    const scalar alpha = 1.f / (_stiffness * dt * dt);
    const scalar dt_lambda = -(C + alpha * _lambda) / (sum_norm_grad + alpha);
    _lambda += dt_lambda;

    for (int i = 0; i < nb(); ++i) {
        x[i]->force += (dt_lambda * x[i]->inv_mass * grads[i]); // we use force to store dt_x
    }
}

scalar XPBD_Constraint::get_dual_residual(const std::vector<Particle*>& particles, const scalar dt) {
    std::vector<Particle*> x(this->nb());
    for (int i = 0; i < this->nb(); ++i) {
        x[i] = particles[ids[i]];
    }
    scalar C = 0;
    std::vector<Vector3> grads(this->nb(), Unit3D::Zero());
    if (!project(x, grads, C)) return 0;
    const scalar alpha = 1.0f / (_stiffness * dt * dt);
    return (C + alpha * _lambda);
}


void XPBD_DistanceConstraint::init(const std::vector<Particle*>& particles) {
    const Vector3 pa = particles[ids[0]]->position;
    const Vector3 pb = particles[ids[1]]->position;
    _rest_length = glm::distance(pa, pb);
}

bool XPBD_DistanceConstraint::project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) {
    Vector3 n = x[0]->position - x[1]->position;
    const scalar d = glm::length(n);
    if (d < 1e-6) return false;

    n /= d;
    C = d - _rest_length;

    if (std::abs(C) < 1e-6) return false;

    grads[0] = n;
    grads[1] = -n;

    return true;
}