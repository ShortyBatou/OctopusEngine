#include "Dynamic/FEM/FEM_Generic.h"
#include <vector>

FEM_Generic::~FEM_Generic() {
    delete _shape;
    delete _material;
}

Matrix3x3 FEM_Generic::get_jacobian(const std::vector<Particle *> &p, const std::vector<Vector3> &dN) {
    Matrix3x3 J = Matrix::Zero3x3();
    for (size_t j = 0; j < dN.size(); ++j) {
        J += glm::outerProduct(p[j]->position, dN[j]);
    }
    return J;
}

Matrix3x3 FEM_Generic::get_jacobian(const std::vector<Vector3> &p, const std::vector<Vector3> &dN) {
    Matrix3x3 J = Matrix::Zero3x3();
    for (size_t j = 0; j < dN.size(); ++j) {
        J += glm::outerProduct(p[j], dN[j]);
    }
    return J;
}

scalar FEM_Generic::compute_volume(const std::vector<Particle *> &p) const {
    size_t nb_w = _shape->weights.size();
    size_t nb_s = _shape->dN.size();
    assert(p.size() >= nb_s);
    scalar volume = 0.;
    for (size_t i = 0; i < nb_w; ++i) {
        scalar d = glm::determinant(get_jacobian(p, _shape->dN[i]));
        volume += abs(d) * _shape->weights[i];
    }
    return volume;
}

scalar FEM_Generic::compute_volume(const std::vector<Vector3> &p) const {
    size_t nb_w = _shape->weights.size();
    size_t nb_s = _shape->dN.size();
    assert(p.size() >= nb_s);
    scalar volume = 0.;
    for (size_t i = 0; i < nb_w; ++i) {
        scalar d = glm::determinant(get_jacobian(p, _shape->dN[i]));
        volume += abs(d) * _shape->weights[i];
    }
    return volume;
}

scalar FEM_Generic::compute_stress(const std::vector<Vector3> &p) const {
    size_t nb_w = _shape->weights.size();
    size_t nb_s = _shape->dN.size();
    assert(p.size() >= nb_s);
    scalar stress = 0.;
    for (size_t i = 0; i < nb_w; ++i) {
        Matrix3x3 F = get_jacobian(p, _shape->dN[i]) * _JX_inv[i];
        Matrix3x3 P = _material->get_pk1(F);
        P = ContinuousMaterial::pk1_to_chauchy_stress(F, P);
        stress += ContinuousMaterial::von_mises_stress(P) * _V[i];
    }
    return stress;
}


// particles : element's particles only and in the right order
void FEM_Generic::build(const std::vector<Particle *> &p) {
    _V.resize(_shape->weights.size());
    _JX_inv.resize(_shape->weights.size());
    _init_volume = 0;
    Matrix3x3 J;
    for (int i = 0; i < _shape->weights.size(); ++i) {
        Matrix3x3 J = get_jacobian(p, _shape->dN[i]);
        _V[i] = abs(glm::determinant(J)) * _shape->weights[i];
        _init_volume += _V[i];
        _JX_inv[i] = glm::inverse(J);
    }
}