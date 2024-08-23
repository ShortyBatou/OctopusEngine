#include "Dynamic/FEM/ContinuousMaterial.h"
#include <string>

std::string get_material_name(Material material) {
    switch (material) {
        case Hooke: return "Hooke";
        case StVK: return "SaintVenant";
        case NeoHooke: return "Neo-Hooke";
        case Stable_NeoHooke: return "Stable Neo-Hooke";
        default: return "";
    }
}


Matrix3x3 ContinuousMaterial::get_strain_linear(const Matrix3x3 &F) {
    return scalar(0.5) * (glm::transpose(F) + F) - Matrix::Identity3x3();
}

Matrix3x3 ContinuousMaterial::get_strain_tensor(const Matrix3x3 &F) {
    return scalar(0.5) * (glm::transpose(F) * F - Matrix::Identity3x3());
}


Matrix3x3 ContinuousMaterial::pk1_to_chauchy_stress(Matrix3x3 &F, Matrix3x3 &P) {
    return P * glm::transpose(F) * (1.f / glm::determinant(F));
}

Matrix3x3 ContinuousMaterial::chauchy_to_PK1_stress(Matrix3x3 &F, Matrix3x3 &C) {
    return glm::determinant(F) * C * glm::transpose(glm::inverse(F));
}

Matrix3x3 ContinuousMaterial::chauchy_to_PK2_stress(Matrix3x3 &F, Matrix3x3 &C) {
    Matrix3x3 F_inv = glm::inverse(F);
    return glm::determinant(F) * F_inv * C * glm::transpose(F_inv);
}

// general von misses stress
scalar ContinuousMaterial::von_mises_stress(Matrix3x3 &C) {
    const scalar s0_1 = C[0][0] - C[1][1];
    const scalar s1_2 = C[1][1] - C[2][2];
    const scalar s0_2 = C[0][0] - C[2][2];
    scalar s = 0.5f * (s0_1 * s0_1 + s1_2 * s1_2 + s0_2 * s0_2);
    s += 3.f * (C[0][1] * C[0][1] + C[1][2] * C[1][2] + C[0][2] * C[0][2]);
    return sqrt(s);
}
