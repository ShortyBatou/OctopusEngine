#pragma once
#include "Dynamic/FEM/ContinuousMaterial.h"
#include <vector>

struct FEM_ContinuousMaterial : ContinuousMaterial {
    FEM_ContinuousMaterial(const scalar _young, const scalar _poisson) : ContinuousMaterial(_young, _poisson) { }
    virtual void get_sub_hessian(const Matrix3x3&, std::vector<Matrix3x3>&) { };
};

struct M_Hooke : public FEM_ContinuousMaterial {
    M_Hooke(const scalar _young, const scalar _poisson) : FEM_ContinuousMaterial(_young, _poisson) { }

    Matrix3x3 get_pk1(const Matrix3x3& F) override;

    scalar get_energy(const Matrix3x3& F) override;

    // get all H_kl of dF_i**T H_kl dF_j
    virtual std::vector<Matrix3x3> get_sub_hessian(const Matrix3x3&);
};


struct M_StVK : FEM_ContinuousMaterial {
    M_StVK(const scalar _young, const scalar _poisson) : FEM_ContinuousMaterial(_young, _poisson) { }

    Matrix3x3 get_pk1(const Matrix3x3& F) override;

    scalar get_energy(const Matrix3x3& F) override;
};

struct M_NeoHooke : public FEM_ContinuousMaterial {
    M_NeoHooke(const scalar _young, const scalar _poisson) : FEM_ContinuousMaterial(_young, _poisson) {
    }

    Matrix3x3 get_pk1(const Matrix3x3& F) override;

    scalar get_energy(const Matrix3x3& F) override;
};


struct M_Stable_NeoHooke : public FEM_ContinuousMaterial {
    scalar alpha;
    M_Stable_NeoHooke(const scalar _young, const scalar _poisson) : FEM_ContinuousMaterial(_young, _poisson) {
        alpha = 1 + this->mu / this->lambda;
    }

    Matrix3x3 get_pk1(const Matrix3x3& F) override;
    scalar get_energy(const Matrix3x3& F) override;
};

FEM_ContinuousMaterial* get_fem_material(Material material, scalar young, scalar poisson);