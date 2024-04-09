#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/Base/Constraint.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include <vector>
class FEM_Generic : public Constraint {
public:
    FEM_Generic(int* ids, FEM_ContinuousMaterial* material, FEM_Shape* shape)
        : Constraint(std::vector<int>(ids, ids + shape->nb)), _material(material), _shape(shape)
    {  }

    virtual void init(const std::vector<Particle*>& particles) override {
        std::vector<Vector3> X(_shape->nb);
        for (int i = 0; i < _shape->nb; ++i) {
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
        for (int i = 0; i < _weights.size(); ++i) {
            s = coords[i * 3]; t = coords[i * 3 + 1]; l = coords[i * 3 + 2];
            _dN[i] = _shape->build_shape_derivatives(s, t, l);
            _JX[i] = Matrix::Zero3x3();
            for (int j = 0; j < _shape->nb; ++j) {
                _JX[i] += glm::outerProduct(X[j], _dN[i][j]);
            }
            _V[i] = abs(glm::determinant(_JX[i])) * _weights[i];
            init_volume +=  _V[i];

            _JX_inv[i] = glm::inverse(_JX[i]);

            _dF[i].resize(_dN[i].size());
            for (int j = 0; j < _shape->nb; ++j) {
                _dF[i][j] = glm::transpose(_JX_inv[i]) * _dN[i][j];
            }
        }

        volume = init_volume;
    }

    virtual void apply(const std::vector<Particle*>& particles, const scalar) override 
    {
        Matrix3x3 Jx, F, P;

        std::vector<Particle*> p(_shape->nb);
        for (int i = 0; i < _shape->nb; ++i) {
            p[i] = particles[this->_ids[i]];
        }
        volume = 0;
        for (int i = 0; i < _weights.size(); ++i) {
            Jx = Matrix::Zero3x3();
            for (int j = 0; j < _shape->nb; ++j) {
                Jx += glm::outerProduct(p[j]->position, _dN[i][j]);
            }

            volume += abs(glm::determinant(Jx)) * _weights[i];

            F = Jx * _JX_inv[i];
            _material->getStressTensor(F, P);
            for (int j = 0; j < _shape->nb; ++j) {
                p[j]->force -= P * _dF[i][j] * _V[i];
            }
        }
    }

    virtual void compute_df_ij(const Vector3 dFi, const std::vector<Vector3>& Hkl_dFj, Matrix3x3& df_ij) {
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                df_ij[i][j] = glm::dot(dFi, Hkl_dFj[i + j * 3]);
    }

    virtual void pre_compute_hessian(const Vector3& dF_j, const std::vector<Matrix3x3>& H_kl, std::vector<Vector3>& Hkl_dFj) {
        for (int l = 0; l < 3; ++l)
            for (int k = 0; k < 3; ++k)
                Hkl_dFj[k + l * 3] = H_kl[k + l * 3] * dF_j;

    }

    // assemble hessian for one integration point, just for testing
    virtual bool solve_dforces(const Matrix3x3& F, const std::vector<Vector3>& dF_w) {
        // ours
        std::vector<Matrix3x3> H_kl; _material->getSubHessians(F, H_kl);
        std::vector<Vector3> Hkl_dFj(9);
        std::vector<Matrix3x3> df(_shape->nb * _shape->nb);
        for (int j = 0; j < _shape->nb; ++j) {
            pre_compute_hessian(dF_w[j], H_kl, Hkl_dFj);
            for (int i = 0; i < _shape->nb; ++i) {
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

    FEM_ContinuousMaterial* _material;
    FEM_Shape* _shape;
};


class FEM_SVD_Generic : public Constraint {
public:
    FEM_SVD_Generic(int* ids, SVD_ContinuousMaterial* material, FEM_Shape* shape)
        : Constraint(std::vector<int>(ids, ids + shape->nb)), _material(material), _shape(shape)
    {  }

    virtual void init(const std::vector<Particle*>& particles) override {
        std::vector<Vector3> X(this->nb());
        for (int i = 0; i < X.size(); ++i) {
            X[i] = particles[this->_ids[i]]->position;
        }

        int nb_quadrature = _shape->weights.size();
        _V.resize(nb_quadrature);
        _JX_inv.resize(nb_quadrature);
        init_volume = 0;
        Matrix3x3 JX;
        for (int i = 0; i < nb_quadrature; ++i) {
            JX = Matrix::Zero3x3();
            for (int j = 0; j < this->nb(); ++j) {
                JX += glm::outerProduct(X[j], _shape->dN[i][j]);
            }
            _V[i] = std::abs(glm::determinant(JX)) * _shape->weights[i];
            init_volume += _V[i];
            _JX_inv[i] = glm::inverse(JX);
        }
        volume = init_volume;
    }

    virtual void apply(const std::vector<Particle*>& particles, const scalar) override
    {
        Matrix3x3 Jx, F;
        Matrix3x3 U, S, V;
        Matrix3x3 W[3];
        Vector3 constraint;

        std::vector<Particle*> x(_shape->nb);
        for (int i = 0; i < _shape->nb; ++i) {
            x[i] = particles[this->_ids[i]];
        }

        int nb_quadrature = _shape->weights.size();
        for (int i = 0; i < nb_quadrature; ++i) {
            Jx = Matrix::Zero3x3();
            for (int j = 0; j < this->nb(); ++j) {
                Jx += glm::outerProduct(x[j]->position, _shape->dN[i][j]);
            }

            F = Jx * _JX_inv[i];

            // get SVD of F
            MatrixAlgo::SVD(F, U, S, V);

            Vector3 s = Vector3(S[0][0], S[1][1], S[2][2]);

            // get constraint
            _material->getConstraint(s, constraint);
   
            for (int j = 0; j < 3; ++j) {
                W[j] = glm::outerProduct(U[j], V[j]) * glm::transpose(_JX_inv[i]);
            }

            // get grads 
            for (int j = 0; j < _shape->nb; j++) {
                Matrix3x3 J = Matrix::Zero3x3();
                J[0] = W[0] * _shape->dN[i][j];
                J[1] = W[1] * _shape->dN[i][j];
                J[2] = W[2] * _shape->dN[i][j];
                x[j]->force -= J * constraint * V[i];
            }

        }
    }

    scalar get_volume() { return volume; }

    scalar init_volume;
    scalar volume;
private:
    std::vector<std::vector<Vector3>> _dN;
    std::vector<Matrix3x3> _JX_inv;
    std::vector<scalar> _V;

    SVD_ContinuousMaterial* _material;
    FEM_Shape* _shape;
};
