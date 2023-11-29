#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinousMaterial.h"
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
        std::vector<scalar> coords = _shape->getQuadratureCoordinates();
        _weights = _shape->getWeights();

        _dN.resize(_weights.size());
        _detJ.resize(_weights.size());
        _JX_inv.resize(_weights.size());
        _JX.resize(_weights.size());
        init_volume = 0;
        for (unsigned int i = 0; i < _weights.size(); ++i) {
            s = coords[i * 3]; t = coords[i * 3 + 1]; l = coords[i * 3 + 2];
            _dN[i] = _shape->build_shape_derivatives(s, t, l);
            _JX[i] = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                _JX[i] += glm::outerProduct(X[j], _dN[i][j]);
            }
            _detJ[i] = std::abs(glm::determinant(_JX[i]));
            init_volume += _detJ[i] * _weights[i];
            _JX_inv[i] = glm::inverse(_JX[i]);
        }
        volume = init_volume;
    }


    virtual bool  project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Matrix3x3 Jx, F, P;
        scalar energy;
        volume = 0;
        for (unsigned int i = 0; i < _weights.size(); ++i) {
            Jx = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                Jx += glm::outerProduct(x[j]->position, _dN[i][j]);
            }

            volume += std::abs(glm::determinant(Jx)) * _weights[i];

            F = Jx * _JX_inv[i];
            _material->getStressTensorAndEnergy(F, P, energy);

            P = P * glm::transpose(_JX_inv[i]) * _detJ[i] * _weights[i];
            for (unsigned int j = 0; j < this->nb(); ++j) {
                grads[j] += P * _dN[i][j];
            }

            C += energy * _detJ[i] * _weights[i];
        }


        if (std::abs(C) <= scalar(1e-8)) return false;
        scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
        C = sqrt(abs(C));
        scalar C_inv = scalar(1.) / scalar(2. * C);
        for (unsigned int j = 0; j < this->nb(); ++j) {
            grads[j] *= C_inv * s;
        }

        return true;
    }

    void draw_debug(const std::vector<Particle*>& particles) override {
        std::vector<Vector3> pts(this->nb());
        for (unsigned int i = 0; i < this->nb(); ++i) {
            pts[i] = particles[this->_ids[i]]->position;
        }
        _shape->debug_draw(pts);
    }
    scalar init_volume;
    scalar volume;
protected:
    std::vector<std::vector<Vector3>> _dN;
    std::vector<Matrix3x3> _JX_inv;
    std::vector<Matrix3x3> _JX;

    std::vector<scalar> _detJ;
    std::vector<scalar> _weights;

    PBD_ContinuousMaterial* _material;
    FEM_Shape* _shape;
};
