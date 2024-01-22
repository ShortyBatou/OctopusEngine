#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/Base/Constraint.h"
#include "Dynamic/FEM/ContinousMaterial.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include <vector>
class FEM_Generic : public Constraint {
public:
    FEM_Generic(unsigned int* ids, ContinuousMaterial* material, FEM_Shape* shape) 
        : Constraint(std::vector<unsigned int>(ids, ids + shape->nb)), _material(material), _shape(shape)
    {  }

    virtual void init(const std::vector<Particle*>& particles) override {
        std::vector<Vector3> X(_shape->nb);
        for (unsigned int i = 0; i < _shape->nb; ++i) {
            X[i] = particles[this->_ids[i]]->position;
        }

        scalar s, t, l;
        std::vector<scalar> coords = _shape->get_quadrature_coordinates();
        _weights = _shape->get_weights();

        _dN.resize(_weights.size());
        _dF.resize(_weights.size());
        _V.resize(_weights.size());
        _JX_inv.resize(_weights.size());
        _JX.resize(_weights.size());
        init_volume = 0;
        for (unsigned int i = 0; i < _weights.size(); ++i) {
            s = coords[i * 3]; t = coords[i * 3 + 1]; l = coords[i * 3 + 2];
            _dN[i] = _shape->build_shape_derivatives(s, t, l);
            _JX[i] = Matrix::Zero3x3();
            for (unsigned int j = 0; j < _shape->nb; ++j) {
                _JX[i] += glm::outerProduct(X[j], _dN[i][j]);
            }
            _V[i] = abs(glm::determinant(_JX[i])) * _weights[i];
            init_volume +=  _V[i];

            _JX_inv[i] = glm::inverse(_JX[i]);

            _dF[i].resize(_dN[i].size());
            for (unsigned int j = 0; j < _shape->nb; ++j) {
                _dF[i][j] = glm::transpose(_JX_inv[i]) * _dN[i][j];
            }
        }

        volume = init_volume;
    }

    virtual void apply(const std::vector<Particle*>& particles, const scalar) override 
    {
        Matrix3x3 Jx, F, P;

        std::vector<Particle*> p(_shape->nb);
        for (unsigned int i = 0; i < _shape->nb; ++i) {
            p[i] = particles[this->_ids[i]];
        }
        volume = 0;
        for (unsigned int i = 0; i < _weights.size(); ++i) {
            Jx = Matrix::Zero3x3();
            for (unsigned int j = 0; j < _shape->nb; ++j) {
                Jx += glm::outerProduct(p[j]->position, _dN[i][j]);
            }

            volume += abs(glm::determinant(Jx)) * _weights[i];

            F = Jx * _JX_inv[i];
            _material->getStressTensor(F, P);
            for (unsigned int j = 0; j < _shape->nb; ++j) {
                p[j]->force -= P * _dF[i][j] * _V[i];
            }
        }
    }

    virtual void compute_df_ij(const Vector3 dFi, const std::vector<Vector3>& Hkl_dFj, Matrix3x3& df_ij) {
        for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int i = 0; i < 3; ++i)
                df_ij[i][j] = glm::dot(dFi, Hkl_dFj[i + j * 3]);
    }

    virtual void pre_compute_hessian(const Vector3& dF_j, const std::vector<Matrix3x3>& H_kl, std::vector<Vector3>& Hkl_dFj) {
        for (unsigned int l = 0; l < 3; ++l)
            for (unsigned int k = 0; k < 3; ++k)
                Hkl_dFj[k + l * 3] = H_kl[k + l * 3] * dF_j;

    }

    // assemble hessian for one integration point, just for testing
    virtual bool solve_dforces(const Matrix3x3& F, const std::vector<Vector3>& dF_w) {
        // ours
        std::vector<Matrix3x3> H_kl; _material->getSubHessians(F, H_kl);
        std::vector<Vector3> Hkl_dFj(9);
        std::vector<Matrix3x3> df(_shape->nb * _shape->nb);
        for (unsigned int j = 0; j < _shape->nb; ++j) {
            pre_compute_hessian(dF_w[j], H_kl, Hkl_dFj);
            for (unsigned int i = 0; i < _shape->nb; ++i) {
                compute_df_ij(dF_w[i], Hkl_dFj, df[i + j * _shape->nb]);
            }
        }
        return true;
    }

    scalar get_volume() { return volume; }

    scalar init_volume;
    scalar volume;
private:
    std::vector<std::vector<Vector3>> _dN;
    std::vector<std::vector<Vector3>> _dF;
    std::vector<Matrix3x3> _JX_inv;
    std::vector<Matrix3x3> _JX;
    std::vector<scalar> _V;
    std::vector<scalar> _weights;

    ContinuousMaterial* _material;
    FEM_Shape* _shape;
};
