#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"

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
        JX[0] = X[0] - X[3];
        JX[1] = X[1] - X[3];
        JX[2] = X[2] - X[3];

        _V_init = std::abs(glm::determinant(JX)) / scalar(6);
        _JX_inv = glm::inverse(JX);
        this->_stiffness = _material->getStiffness() * _V_init;
        _V = _V_init;
    }


    virtual bool  project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Matrix3x3 Jx = Matrix::Zero3x3(), F, P;
        Jx[0] = x[0]->position - x[3]->position;
        Jx[1] = x[1]->position - x[3]->position;
        Jx[2] = x[2]->position - x[3]->position;

        F = Jx * _JX_inv;
        _material->getStressTensorAndEnergy(F, P, C);
        _V = std::abs(glm::determinant(Jx)) / scalar(6);

        P = P * glm::transpose(_JX_inv);

        grads[0] = P[0];
        grads[1] = P[1];
        grads[2] = P[2];
        grads[3] = -grads[0] - grads[1] - grads[2];

        if (std::abs(C) <= scalar(1e-12)) return false;

        return true;
    }

    scalar get_volume() {
        return _V;
    }

    scalar _V_init;
    scalar _V;
protected:
    Matrix3x3 _JX_inv;
   
    PBD_ContinuousMaterial* _material;
};