#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/Base/Constraint.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_Generic.h"
#include <vector>

class FEM_Generic_Force final : public Constraint, public FEM_Generic {
public:
    FEM_Generic_Force(const std::vector<int> &ids, FEM_ContinuousMaterial *material, FEM_Shape *shape)
        : Constraint(ids), FEM_Generic(material, shape), _fem_material(material) {
    }

    void init(const std::vector<Particle *> &particles) override;

    void apply(const std::vector<Particle *> &particles, scalar) override;

    void compute_df_ij(Vector3 dFi, const std::vector<Vector3> &Hkl_dFj, Matrix3x3 &df_ij);

    void pre_compute_hessian(const Vector3 &dF_j, const std::vector<Matrix3x3> &H_kl,
                             std::vector<Vector3> &Hkl_dFj);

    // assemble hessian for one integration point, just for testing
    bool solve_dforces(const Matrix3x3 &F, const std::vector<Vector3> &dF_w);

private:
    FEM_ContinuousMaterial *_fem_material;
    std::vector<std::vector<Vector3> > _dF;
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
