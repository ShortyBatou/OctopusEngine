#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinousMaterial.h"

class XPBD_FEM_Tetra : public XPBD_Constraint {
public:
    XPBD_FEM_Tetra(unsigned int* ids, PBD_ContinuousMaterial* material)
        : XPBD_Constraint(std::vector<unsigned int>(ids, ids + 4), material->getStiffness()), _material(material)
    { }

    virtual void init(const std::vector<Particle*>& particles) override {
        std::vector<Vector3> X(this->nb());
        for (unsigned int i = 0; i < X.size(); ++i) {
            X[i] = particles[this->_ids[i]]->position;
        }

        Matrix3x3 JX = Matrix::Zero3x3();
        JX[0] = X[1] - X[0];
        JX[1] = X[2] - X[0];
        JX[2] = X[3] - X[0];

        _V = std::abs(glm::determinant(JX)) / scalar(6);
        _JX_inv = glm::inverse(JX);
        this->_stiffness = _material->getStiffness() * _V;
    }


    virtual bool  project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Matrix3x3 Jx = Matrix::Zero3x3(), F, P;
        Jx[0] = x[1]->position - x[0]->position;
        Jx[1] = x[2]->position - x[0]->position;
        Jx[2] = x[3]->position - x[0]->position;

        F = Jx * _JX_inv;
        _material->getStressTensorAndEnergy(F, P, C);


        P = P * glm::transpose(_JX_inv);

        grads[1] = P[0];
        grads[2] = P[1];
        grads[3] = P[2];
        grads[0] = -grads[1] - grads[2] - grads[3];

        // temp
        _Vx = std::abs(glm::determinant(Jx)) / scalar(6);

        if (std::abs(C) <= scalar(1e-8)) return false;

        return true;
    }

protected:
    Matrix3x3 _JX_inv;
    scalar _V, _Vx;

    PBD_ContinuousMaterial* _material;
};