#pragma once
#include "Core/Base.h"
#include <vector>
enum Material {
    Hooke, StVK, NeoHooke, Stable_NeoHooke
};

std::string get_material_name(Material material) {
    switch (material)
    {
        case Hooke: return "Hooke";
        case StVK: return "SaintVenant";
        case NeoHooke: return "Neo-Hooke";
        case Stable_NeoHooke: return "Stable Neo-Hooke";
        default: return "";
    }
}

struct ContinuousMaterial {
    scalar lambda, mu;
    scalar young, poisson;
    

    ContinuousMaterial(const scalar _young, const scalar _poisson) : young(_young), poisson(_poisson) {
        lambda = compute_lambda();
        mu = compute_mu();
    }
    scalar compute_lambda() const { return scalar(young * poisson / ((1. + poisson) * (1. - 2. * poisson))); }
    scalar compute_mu() const { return scalar(young / (2. * (1. + poisson))); }

    virtual scalar get_energy(const Matrix3x3& F) = 0;
    virtual Matrix3x3 get_pk1(const Matrix3x3& F) = 0;

    static Matrix3x3 get_strain_linear(const Matrix3x3& F)
    {
        return scalar(0.5) * (glm::transpose(F) + F) - Matrix::Identity3x3();
    }

    static Matrix3x3 get_strain_tensor(const Matrix3x3& F)
    {
        return scalar(0.5) * (glm::transpose(F) * F - Matrix::Identity3x3());
    }


    static Matrix3x3 pk1_to_chauchy_stress(Matrix3x3& F, Matrix3x3& P) {
        return P * glm::transpose(F) * (1.f / glm::determinant(F));
    }

    static Matrix3x3 chauchy_to_PK1_stress(Matrix3x3& F, Matrix3x3& C) {
        return glm::determinant(F) * C * glm::transpose(glm::inverse(F));
    }

    static Matrix3x3 chauchy_to_PK2_stress(Matrix3x3& F, Matrix3x3& C) {
        Matrix3x3 F_inv = glm::inverse(F);
        return glm::determinant(F) * F_inv * C * glm::transpose(F_inv);
    }

    // general von misses stress
    static scalar von_mises_stress(Matrix3x3& C) {
        scalar s0_1 = C[0][0] - C[1][1];
        scalar s1_2 = C[1][1] - C[2][2];
        scalar s0_2 = C[0][0] - C[2][2];
        scalar s = 0.5f * (s0_1*s0_1 + s1_2*s1_2 + s0_2*s0_2);
        s += 3. * (C[0][1] * C[0][1] + C[1][2] * C[1][2] + C[0][2] * C[0][2]);
        return sqrt(s);
    }

    virtual ~ContinuousMaterial() {}
};

