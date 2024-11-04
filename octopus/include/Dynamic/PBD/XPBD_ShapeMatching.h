#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"
#include "Manager/Debug.h"
class XPBD_ShapeMatching final : public XPBD_Constraint {
public:
    XPBD_ShapeMatching(const std::vector<int> &ids, PBD_ContinuousMaterial *material, const scalar volume)
        : XPBD_Constraint(ids, material->get_stiffness()), _total_mass(0), _com_init(0),
          _JX_inv(Matrix::Zero3x3()), _V(volume), _material(material) {
    }

    void init(const std::vector<Particle *> &particles) override {
        _total_mass = 0;
        std::vector<Particle *> parts;
        parts.reserve(nb());
        for (const int id: ids) parts.push_back(particles[id]);

        for (int i = 0; i < nb(); ++i) _total_mass += parts[i]->mass;

        _com_init = compute_com(parts);

        _r_init.resize(nb());
        for (int i = 0; i < nb(); ++i)
            _r_init[i] = parts[i]->position - _com_init;

        _JX_inv = glm::inverse(compute_deform(parts, _com_init));
    }

    [[nodiscard]] Vector3 compute_com(const std::vector<Particle *> &particles) const {
        assert(particles.size() == nb());
        Vector3 com = Unit3D::Zero();
        for (const Particle *parts: particles) com += parts->position * parts->mass;
        com /= _total_mass;
        return com;
    }

    [[nodiscard]] Matrix3x3 compute_deform(const std::vector<Particle *> &particles, const Vector3 com) const {
        assert(particles.size() == nb());
        Matrix3x3 J = Matrix::Zero3x3();
        for (int i = 0; i < nb(); ++i) {
            Vector3 ri = particles[i]->position - com;
            J += particles[i]->mass * glm::outerProduct(ri, _r_init[i]);
        }
        return J;
    }


    bool project(const std::vector<Particle *> &particles, std::vector<Vector3> &grads, scalar &C) override {
        const Vector3 com = compute_com(particles);
        const Matrix3x3 Jx = compute_deform(particles, com);
        const Matrix3x3 F = Jx * _JX_inv;
        Matrix3x3 P = Matrix::Zero3x3();
        _material->get_pk1_and_energy(F, P, C);

        P *= glm::transpose(_JX_inv) * _V;
        for (int i = 0; i < nb(); ++i) {
            grads[i] = particles[i]->mass * P * _r_init[i];
        }

        C *= _V;
        C = sqrt(abs(C));
        if (C < eps) return false;
        const scalar C_inv = 0.5f / C;
        for (int i = 0; i < nb(); ++i) {
            grads[i] *= C_inv;
        }
        return true;
    }

    ~XPBD_ShapeMatching() override {
        delete _material;
    }

    [[nodiscard]] Matrix3x3 get_init_deform() const {
        return _JX_inv;
    }

    [[nodiscard]] Vector3 get_r_init(int i) const {
        return _r_init[i];
    }

protected:
    Matrix3x3 _JX_inv;
    scalar _V;
    scalar _total_mass;
    Vector3 _com_init;
    std::vector<Vector3> _r_init;
    PBD_ContinuousMaterial *_material;
};

class XPBD_ShapeMatching_Filtering final : public XPBD_Constraint {
public:
    explicit XPBD_ShapeMatching_Filtering(XPBD_ShapeMatching *sm)
        : XPBD_Constraint(sm->ids, std::numeric_limits<scalar>::infinity()), _sm(sm) {
    }

    void apply(const std::vector<Particle *> &particles, const scalar dt) override {
        if (_stiffness <= 0) return;

        std::vector<Particle *> x(nb());
        for (int i = 0; i < nb(); ++i) {
            x[i] = particles[ids[i]];
        }
        const Vector3 com = _sm->compute_com(x);
        const Matrix3x3 F = _sm->compute_deform(x, com) * _sm->get_init_deform();
        for (int i = 0; i < nb(); ++i) {
            x[i]->position = com + F * _sm->get_r_init(i);
        }
        Debug::SetColor(ColorBase::Red());
        Debug::Cube(com, 0.03);
    }

    bool project(const std::vector<Particle *> &particles, std::vector<Vector3> &grads, scalar &C) override {
        return true;
    }

protected:
    XPBD_ShapeMatching *_sm;
};
