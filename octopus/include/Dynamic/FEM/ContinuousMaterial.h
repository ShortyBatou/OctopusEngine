#pragma once
#include "Core/Base.h"
#include <string>

enum Material {
    Hooke, StVK, NeoHooke, Stable_NeoHooke
};

std::string get_material_name(Material material);

struct ContinuousMaterial {
    scalar lambda, mu;
    scalar young, poisson;

    ContinuousMaterial(const scalar _young, const scalar _poisson) : young(_young), poisson(_poisson) {
        lambda = compute_lambda();
        mu = compute_mu();
    }

    [[nodiscard]] scalar compute_lambda() const { return scalar(young * poisson / ((1.f + poisson) * (1.f - 2.f * poisson))); }
    [[nodiscard]] scalar compute_mu() const { return scalar(young / (2.f * (1.f + poisson))); }

    virtual scalar get_energy(const Matrix3x3 &F) = 0;

    virtual Matrix3x3 get_pk1(const Matrix3x3 &F) = 0;

    static Matrix3x3 get_strain_linear(const Matrix3x3 &F);

    static Matrix3x3 get_strain_tensor(const Matrix3x3 &F);

    static Matrix3x3 pk1_to_chauchy_stress(Matrix3x3 &F, Matrix3x3 &P);

    static Matrix3x3 chauchy_to_PK1_stress(Matrix3x3 &F, Matrix3x3 &C);

    static Matrix3x3 chauchy_to_PK2_stress(Matrix3x3 &F, Matrix3x3 &C);

    static scalar von_mises_stress(Matrix3x3 &C);

    virtual ~ContinuousMaterial() = default;
};
