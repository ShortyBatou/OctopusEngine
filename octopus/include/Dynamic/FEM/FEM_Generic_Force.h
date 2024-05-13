#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/Base/Constraint.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_Generic.h"
#include <vector>
#include <Tools/MatrixAlgo.h>

class FEM_Generic_Force : public Constraint, public FEM_Generic {
public:
    FEM_Generic_Force(const std::vector<int>& ids, FEM_ContinuousMaterial* material, FEM_Shape* shape)
        : Constraint(ids), FEM_Generic(material, shape), _fem_material(material)
    {  }

    virtual void init(const std::vector<Particle*>& particles) override {
        build(get_particles(particles));
        _dF.resize(_shape->weights.size());
        for (int i = 0; i < _shape->weights.size(); ++i) {
            _dF[i].resize(_shape->dN[i].size());
            for (int j = 0; j < _shape->nb; ++j) {
                _dF[i][j] = glm::transpose(_JX_inv[i]) * _shape->dN[i][j];
            }
        }
    }

    virtual void apply(const std::vector<Particle*>& particles, const scalar) override 
    {
        Matrix3x3 Jx, F, P;
        std::vector<Particle*> p = get_particles(particles);
        for (int i = 0; i < _shape->weights.size(); ++i) {
            Jx = get_jacobian(p, _shape->dN[i]);
            F = Jx * _JX_inv[i];
            P = _material->get_pk1(F);
            for (int j = 0; j < _shape->nb; ++j) {
                p[j]->force -= P * _dF[i][j] * _V[i];
            }
        }
        std::cout;
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
        std::vector<Matrix3x3> H_kl; _fem_material->get_sub_hessian(F, H_kl);
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

private:
    FEM_ContinuousMaterial* _fem_material;
    std::vector<std::vector<Vector3>> _dF;
};

/*
class FEM_SVD_Generic : public Constraint, public FEM_Generic {
public:
    FEM_SVD_Generic(int* ids, SVD_ContinuousMaterial* material, FEM_Shape* shape)
        : Constraint(std::vector<int>(ids, ids + shape->nb)), FEM_Generic(material, shape)
    {  }

    virtual void init(const std::vector<Particle*>& particles) override {
        build(get_particles(particles));
    }

    virtual void apply(const std::vector<Particle*>& particles, const scalar) override
    {
        Matrix3x3 Jx, F;
        Matrix3x3 U, S, V;
        Matrix3x3 W[3];
        Vector3 constraint;

        std::vector<Particle*> x = get_particles(particles);

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
            _material->get_constraint(s, constraint);
   
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
};*/
