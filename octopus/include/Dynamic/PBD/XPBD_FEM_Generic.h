#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Shape.h"

class XPBD_FEM_Generic : public XPBD_Constraint {
public:
    XPBD_FEM_Generic(unsigned int* ids, PBD_ContinuousMaterial* material, FEM_Shape* shape)
        : XPBD_Constraint(std::vector<unsigned int>(ids, ids + shape->nb) , material->getStiffness()), _material(material), _shape(shape)
    { }

    virtual void init(const std::vector<Particle*>& particles) override {
        std::vector<Vector3> X(this->nb());
        for (unsigned int i = 0; i < X.size(); ++i) {
            X[i] = particles[this->_ids[i]]->position;
        }

        scalar s, t, l;
        unsigned int nb_quadrature = _shape->weights.size();
        _V.resize(nb_quadrature);
        _JX_inv.resize(nb_quadrature);
        
        Matrix3x3 JX;
        init_volume = 0;
        for (unsigned int i = 0; i < nb_quadrature; ++i) {
            JX = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                JX += glm::outerProduct(X[j], _shape->dN[i][j]);
            }
            _V[i] = std::abs(glm::determinant(JX)) * _shape->weights[i];
            init_volume += _V[i];
            _JX_inv[i] = glm::inverse(JX);
        }
    }


    virtual bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Matrix3x3 Jx, F, P;
        scalar energy;
        unsigned int nb_quadrature = _shape->weights.size();
        for (unsigned int i = 0; i < nb_quadrature; ++i) {
            Jx = Matrix::Zero3x3();

            // Compute transform (reference => scene)
            for (unsigned int j = 0; j < this->nb(); ++j) {
                Jx += glm::outerProduct(x[j]->position, _shape->dN[i][j]);
            }

            // Deformation gradient (material => scene   =   material => reference => scene)
            F = Jx * _JX_inv[i];

            // Get piola kirchoff stress tensor + energy
            _material->getStressTensorAndEnergy(F, P, energy);

            // add forces
            P = P * glm::transpose(_JX_inv[i]) * _V[i];
            for (unsigned int j = 0; j < this->nb(); ++j) {
                grads[j] += P * _shape->dN[i][j];
            }

            // add energy
            C += energy * _V[i];
        }

        // convert energy to constraint
        if (std::abs(C) <= scalar(1e-24)) return false;
        scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
        C = sqrt(abs(C)) * s;

        // convert force to constraint gradient
        scalar C_inv = scalar(1.) / scalar(2. * C);
        for (unsigned int j = 0; j < this->nb(); ++j) {
            grads[j] *= C_inv;
        }
        return true;
    } 

    scalar get_init_volume() { return init_volume; }

    virtual ~XPBD_FEM_Generic() {
        delete _material;
        delete _shape;
    }

public:
    scalar init_volume;
    std::vector<Matrix3x3> _JX_inv;
    std::vector<scalar> _V;

    // it's soposed to be static
    PBD_ContinuousMaterial* _material;
    FEM_Shape* _shape;
};