#pragma once
#include "Dynamic/PBD/XPBD_FEM_Tetra.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"


void XPBD_FEM_Tetra::init(const std::vector<Particle *> &particles) {
    std::vector<Vector3> X(this->nb());
    for (int i = 0; i < X.size(); ++i) {
        X[i] = particles[this->ids[i]]->position;
    }

    Matrix3x3 JX = Matrix::Zero3x3();
    JX[0] = X[0] - X[3];
    JX[1] = X[1] - X[3];
    JX[2] = X[2] - X[3];

    V_init = std::abs(glm::determinant(JX)) / 6.f;
    JX_inv = glm::inverse(JX);
    this->_stiffness = material->get_stiffness() * V_init;
    V = V_init;
}


bool XPBD_FEM_Tetra::project(const std::vector<Particle *> &x, std::vector<Vector3> &grads, scalar &C) {
    Matrix3x3 Jx = Matrix::Zero3x3(), P;
    Jx[0] = x[0]->position - x[3]->position;
    Jx[1] = x[1]->position - x[3]->position;
    Jx[2] = x[2]->position - x[3]->position;

    const Matrix3x3 F = Jx * JX_inv;
    material->get_pk1_and_energy(F, P, C);
    V = std::abs(glm::determinant(Jx)) / 6.f;

    P = P * glm::transpose(JX_inv);

    grads[0] = P[0];
    grads[1] = P[1];
    grads[2] = P[2];
    grads[3] = -grads[0] - grads[1] - grads[2];

    if (std::abs(C) <= 1e-12) return false;

    return true;
}
