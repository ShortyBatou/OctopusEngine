#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"

class XPBD_FEM_Tetra : public XPBD_Constraint {
public:
    XPBD_FEM_Tetra(int *ids, PBD_ContinuousMaterial *material)
        : XPBD_Constraint(std::vector<int>(ids, ids + 4), material->get_stiffness()),
        V_init(0), V(0), JX_inv(Matrix::Zero3x3()), material(material) {
    }

    void init(const std::vector<Particle *> &particles) override;


    bool project(const std::vector<Particle *> &x, std::vector<Vector3> &grads, scalar &C) override;

    scalar V_init;
    scalar V;

protected:
    Matrix3x3 JX_inv;

    PBD_ContinuousMaterial *material;
};
