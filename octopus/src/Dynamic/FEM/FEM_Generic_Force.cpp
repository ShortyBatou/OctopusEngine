#include "Dynamic/FEM/FEM_Generic_Force.h"
#include <iostream>

void FEM_Generic_Force::init(const std::vector<Particle *> &particles) {
    build(get_particles(particles));
    _dF.resize(_shape->weights.size());
    for (int i = 0; i < _shape->weights.size(); ++i) {
        _dF[i].resize(_shape->dN[i].size());
        for (int j = 0; j < _shape->nb; ++j) {
            _dF[i][j] = glm::transpose(_JX_inv[i]) * _shape->dN[i][j];
        }
    }
}

void FEM_Generic_Force::apply(const std::vector<Particle *> &particles, const scalar) {
    std::vector<Particle *> p = get_particles(particles);
    for (int i = 0; i < _shape->weights.size(); ++i) {
        Matrix3x3 Jx = get_jacobian(p, _shape->dN[i]);
        Matrix3x3 F = Jx * _JX_inv[i];
        Matrix3x3 P = _material->get_pk1(F);
        for (int j = 0; j < _shape->nb; ++j) {
            p[j]->force -= P * _dF[i][j] * _V[i];
        }
    }
}

void FEM_Generic_Force::compute_df_ij(const Vector3 dFi, const std::vector<Vector3> &Hkl_dFj, Matrix3x3 &df_ij) {
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            df_ij[i][j] = glm::dot(dFi, Hkl_dFj[i + j * 3]);
}

void FEM_Generic_Force::pre_compute_hessian(const Vector3 &dF_j, const std::vector<Matrix3x3> &H_kl,
                                            std::vector<Vector3> &Hkl_dFj) {
    for (int l = 0; l < 3; ++l)
        for (int k = 0; k < 3; ++k)
            Hkl_dFj[k + l * 3] = H_kl[k + l * 3] * dF_j;
}

// assemble hessian for one integration point, just for testing
bool FEM_Generic_Force::solve_dforces(const Matrix3x3 &F, const std::vector<Vector3> &dF_w) {
    // ours
    std::vector<Matrix3x3> H_kl;
    _fem_material->get_sub_hessian(F, H_kl);
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
