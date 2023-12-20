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
        _V.resize(_weights.size());
        _JX_inv.resize(_weights.size());
        Matrix3x3 JX;
        init_volume = 0;
        for (unsigned int i = 0; i < _weights.size(); ++i) {
            s = coords[i * 3]; t = coords[i * 3 + 1]; l = coords[i * 3 + 2];
            _dN[i] = _shape->build_shape_derivatives(s, t, l);
            JX = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                JX += glm::outerProduct(X[j], _dN[i][j]);
            }
            _V[i] = std::abs(glm::determinant(JX)) * _weights[i];
            init_volume += _V[i];
            _JX_inv[i] = glm::inverse(JX);
        }
        volume = init_volume;
    }


    virtual bool  project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Matrix3x3 Jx, F, P;
        scalar energy;
        volume = 0;
        for (unsigned int i = 0; i < _V.size(); ++i) {
            Jx = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                Jx += glm::outerProduct(x[j]->position, _dN[i][j]);
            }

            volume += std::abs(glm::determinant(Jx)) * _weights[i]; // temp, just for testing

            F = Jx * _JX_inv[i];
            _material->getStressTensorAndEnergy(F, P, energy);

            P = P * glm::transpose(_JX_inv[i]) * _V[i];
            for (unsigned int j = 0; j < this->nb(); ++j) {
                grads[j] += P * _dN[i][j];
            }

            C += energy * _V[i];
        }


        if (std::abs(C) <= scalar(1e-24)) return false;
        scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
        C = sqrt(abs(C)) ;
        scalar C_inv = scalar(1.) / scalar(2. * C);
        for (unsigned int j = 0; j < this->nb(); ++j) {
            grads[j] *= C_inv;
        }

        return true;
    } 

    scalar get_volume() { return volume; }

    void draw_debug(const std::vector<Particle*>& particles) override {
        std::vector<Vector3> pts(this->nb());
        for (unsigned int i = 0; i < this->nb(); ++i) {
            pts[i] = particles[this->_ids[i]]->position;
        }
        _shape->debug_draw(pts);
    }

    virtual ~XPBD_FEM_Generic() {
        delete _material;
        delete _shape;
    }

    scalar init_volume;
    scalar volume;
protected:
    std::vector<Matrix3x3> _JX_inv;
    std::vector<scalar> _V;

    std::vector<scalar> _weights; // just for test

    // it's soposed to be static
    std::vector<std::vector<Vector3>> _dN;

    PBD_ContinuousMaterial* _material;
    FEM_Shape* _shape;
};


class XPBD_FEM_Generic_V2 : public XPBD_Constraint {
public:
    XPBD_FEM_Generic_V2(unsigned int* ids, std::vector<PBD_ContinuousMaterial*> materials, FEM_Shape* shape) 
        : XPBD_Constraint(std::vector<unsigned int>(ids, ids + shape->nb), 1), _materials(materials), _shape(shape)
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
        _V.resize(_weights.size());
        _JX_inv.resize(_weights.size());
        _F.resize(_weights.size());

        Matrix3x3 JX;
        init_volume = 0;
        for (unsigned int i = 0; i < _weights.size(); ++i) {
            s = coords[i * 3]; t = coords[i * 3 + 1]; l = coords[i * 3 + 2];
            _dN[i] = _shape->build_shape_derivatives(s, t, l);
            JX = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                JX += glm::outerProduct(X[j], _dN[i][j]);
            }
            _V[i] = std::abs(glm::determinant(JX)) * _weights[i];
            init_volume += _V[i];
            _JX_inv[i] = glm::inverse(JX);
        }
        volume = init_volume;

    }
    

    virtual void apply(const std::vector<Particle*>& particles, const scalar dt) override {
        if (_stiffness <= 0) return;

        scalar w_sum(0.);
        std::vector<Particle*> x(this->nb());
        for (unsigned int i = 0; i < this->nb(); ++i) {
            x[i] = particles[this->_ids[i]];
            w_sum += x[i]->inv_mass;
        }
        
        compute_F(x);

        for (PBD_ContinuousMaterial* material : _materials) {
            _material = material;

            scalar C = 0;
            std::vector<Vector3> grads(this->nb(), Unit3D::Zero());
            if (!project(x, grads, C)) return;

            scalar sum_norm_grad = 0;
            for (unsigned int i = 0; i < this->nb(); ++i) {
                sum_norm_grad += glm::dot(grads[i], grads[i]) * x[i]->inv_mass;
            }

            if (sum_norm_grad < 1e-24) {
                std::cout << "sum_norm_grad" << std::endl;
                return;
            }
            scalar alpha = scalar(1.0) / (_material->getStiffness() * dt * dt);
            scalar dt_lambda = -C / (sum_norm_grad + alpha);

            if (std::abs(dt_lambda) < 1e-24) {
                std::cout << "dt_lambda = " << dt_lambda << std::endl;
                return;
            }

            for (unsigned int i = 0; i < this->nb(); ++i) {
                x[i]->add_force(dt_lambda * x[i]->inv_mass * grads[i] * 0.5f) ; // we use force to store dt_x
            }
        }
    }

    virtual void compute_F(const std::vector<Particle*>& x) {
        Matrix3x3 Jx;
        volume = 0;
        for (unsigned int i = 0; i < _V.size(); ++i) {
            Jx = Matrix::Zero3x3();
            for (unsigned int j = 0; j < this->nb(); ++j) {
                Jx += glm::outerProduct(x[j]->position, _dN[i][j]);
            }

            volume += std::abs(glm::determinant(Jx)) * _weights[i]; // temp, just for testing

            _F[i] = Jx * _JX_inv[i];
        }
    }

    virtual bool  project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override {
        Matrix3x3 P;
        scalar energy;
        for (unsigned int i = 0; i < _V.size(); ++i) {
            _material->getStressTensorAndEnergy(_F[i], P, energy);

            P = P * glm::transpose(_JX_inv[i]) * _V[i];
            for (unsigned int j = 0; j < this->nb(); ++j) {
                grads[j] += P * _dN[i][j];
            }

            C += energy * _V[i];
        }

        if (abs(C) < scalar(1e-24)) return false;
        scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
        
        C = sqrt(abs(C));
        scalar C_inv = scalar(1.) / scalar(2. * C);
        for (unsigned int j = 0; j < this->nb(); ++j) {
            grads[j] *= C_inv;
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

    virtual ~XPBD_FEM_Generic_V2() {
        for (PBD_ContinuousMaterial* material : _materials)
            delete material;
        _materials.clear();
        delete _shape;
    }

    scalar init_volume;
    scalar volume;
protected:
    std::vector<Matrix3x3> _JX_inv;
    std::vector<scalar> _V;
    std::vector<Matrix3x3> _F;

    std::vector<scalar> _weights; // just for test

    // it's soposed to be static
    std::vector<std::vector<Vector3>> _dN;

    PBD_ContinuousMaterial* _material;
    std::vector<PBD_ContinuousMaterial*> _materials;
    FEM_Shape* _shape;
};