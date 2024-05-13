#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Generic.h"
#include "Dynamic/FEM/FEM_Shape.h"

class XPBD_FEM_Generic : public XPBD_Constraint, public FEM_Generic {
public:
    XPBD_FEM_Generic(const std::vector<int>& ids, PBD_ContinuousMaterial* material, FEM_Shape* shape)
        : XPBD_Constraint(ids, material->get_stiffness()), FEM_Generic(material, shape), _pbd_material(material)
    { }

    virtual void init(const std::vector<Particle*>& particles) override {
        build(get_particles(particles));
    }

    virtual bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Matrix3x3 Jx, F, P;
        scalar energy;
        int nb_quadrature = _shape->weights.size();
        for (int i = 0; i < nb_quadrature; ++i) {
            // Compute transform (reference => scene)
            Jx = get_jacobian(x, _shape->dN[i]);

            // Deformation gradient (material => scene   =   material => reference => scene)
            F = Jx * _JX_inv[i];

            // Get piola kirchoff stress tensor + energy
            _pbd_material->get_pk1_and_energy(F, P, energy);

            // add forces
            P = P * glm::transpose(_JX_inv[i]) * _V[i];
            for (int j = 0; j < this->nb(); ++j) 
                grads[j] += P * _shape->dN[i][j];

            // add energy
            C += energy * _V[i];
        }

        // convert energy to constraint
        if (std::abs(C) <= eps) return false;
        scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
        C = sqrt(abs(C)) * s;

        // convert force to constraint gradient
        scalar C_inv = scalar(1.) / scalar(2. * C);
        for (int j = 0; j < this->nb(); ++j) {
            grads[j] *= C_inv;
        }
        return true;
    } 

    virtual ~XPBD_FEM_Generic() {
    }

private: 
    PBD_ContinuousMaterial* _pbd_material;
};