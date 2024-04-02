#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/SVD_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Tools/MatrixAlgo.h"

class XPBD_FEM_SVD_Generic : public XPBD_Constraint {
public:
    XPBD_FEM_SVD_Generic(unsigned int* ids, SVD_ContinuousMaterial* material, FEM_Shape* shape)
        : XPBD_Constraint(std::vector<unsigned int>(ids, ids + shape->nb), 0), _material(material), _shape(shape)
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
        init_volume = 0;
        Matrix3x3 JX;
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

    virtual void apply(const std::vector<Particle*>& particles, const scalar dt) override {
        std::vector<Particle*> x(this->nb());
        for (unsigned int i = 0; i < this->nb(); ++i) {
            x[i] = particles[this->_ids[i]];
        }

        Vector3 C = Unit3D::Zero();
        std::vector<Matrix3x3> J(this->nb(), Matrix::Zero3x3());
        Matrix3x3 A = Matrix::Zero3x3();
        if (!project(x, J, C, A)) return;

        Matrix3x3 sum_J = Matrix::Zero3x3();
        for (unsigned int i = 0; i < this->nb(); ++i) {
            sum_J += J[i] * glm::transpose(J[i]) * x[i]->inv_mass;
        }

        
        A /= dt * dt;

        dt_lambda = - (A * C) * glm::inverse(sum_J + A);

        for (unsigned int i = 0; i < this->nb(); ++i) {
            x[i]->force += x[i]->inv_mass * J[i] * dt_lambda;
        }
        std::cout;
    }


    virtual bool project(const std::vector<Particle*>& x, std::vector<Matrix3x3>& grads, Vector3& C, Matrix3x3& A) {
        Matrix3x3 Jx, F, stiffness;
        Matrix3x3 U, S, V;
        Matrix3x3 W[3];
        Vector3 constraint;
        unsigned int nb_quadrature = _shape->weights.size();
        for (unsigned int i = 0; i < nb_quadrature; ++i) {
            Jx = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                Jx += glm::outerProduct(x[j]->position, _shape->dN[i][j]);
            }

            F = Jx * _JX_inv[i];

            // get SVD of F
            MatrixAlgo::SVD(F, U, S, V);

            Vector3 s = Vector3(S[0][0], S[1][1], S[2][2]);

            // get constraint
            _material->getConstraint(s, constraint);
            C += constraint * _V[i];

            // get compliance
            _material->getStiffness(s, stiffness);
            A += stiffness * _V[i];

            for (unsigned int j = 0; j < 3; ++j) {
                W[j] = glm::outerProduct(U[j], V[j]) * glm::transpose(_JX_inv[i]);
            }

            // get grads 
            for (unsigned int j = 0; j < _shape->nb; j++) {
                grads[j][0] += W[0] * _shape->dN[i][j];
                grads[j][1] += W[1] * _shape->dN[i][j];
                grads[j][2] += W[2] * _shape->dN[i][j];
                grads[j] *=  _V[i];
            }

        }
        A = glm::inverse(A); // compliance matrix A = N^-1
        return true;
    }

    virtual scalar get_dual_residual(const std::vector<Particle*>& particles, const scalar dt) {
        std::vector<Particle*> x(this->nb());
        for (unsigned int i = 0; i < this->nb(); ++i) {
            x[i] = particles[this->_ids[i]];
        }
        
        Vector3 C = Unit3D::Zero();
        std::vector<Matrix3x3> J(this->nb(), Matrix::Zero3x3());
        Matrix3x3 A = Matrix::Zero3x3();
        if (!project(x, J, C, A)) return 0;

        A *= scalar(1.0) / (dt * dt);
        return glm::length2(C + A * dt_lambda);

    }


    virtual ~XPBD_FEM_SVD_Generic() {
        delete _material;
        delete _shape;
    }

public:
    
    Vector3 dt_lambda;
    
    scalar init_volume;
    std::vector<Matrix3x3> _JX_inv;
    std::vector<scalar> _V;

    SVD_ContinuousMaterial* _material;
    FEM_Shape* _shape;
};