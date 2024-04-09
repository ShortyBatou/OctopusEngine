#pragma once
#include "Dynamic/FEM/ContinuousMaterial.h"

struct FEM_ContinuousMaterial : public ContinuousMaterial {
    FEM_ContinuousMaterial(const scalar _young, const scalar _poisson) : ContinuousMaterial(_young, _poisson) { }
    virtual void getSubHessians(const Matrix3x3&, std::vector<Matrix3x3>&) { };
    virtual void getStressTensor(const Matrix3x3& F, Matrix3x3& P) = 0;
    virtual scalar getEnergy(const Matrix3x3& F) = 0;
    virtual ~FEM_ContinuousMaterial() {}
};



struct M_Hooke : public FEM_ContinuousMaterial {
    M_Hooke(const scalar _young, const scalar _poisson) : FEM_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensor(const Matrix3x3& F, Matrix3x3& P) {
        const auto E = getStrainTensorLinear(F);
        P = this->lambda * Matrix::Trace(E) * Matrix::Identity3x3() + this->mu * E;
    }

    virtual scalar getEnergy(const Matrix3x3& F) {
        const auto E = getStrainTensorLinear(F);
        const auto trace = Matrix::Trace(E);
        // Psi(F) = lambda / 2 tr(E)² + mu tr(E^2)
        return 0.5f * this->lambda * trace * trace + this->mu * Matrix::SquaredNorm(E);
    }

    // get all H_kl of dF_i^T H_kl dF_j
    virtual void getSubHessians(const Matrix3x3&, std::vector<Matrix3x3>& H) {
        H.resize(9, Matrix::Zero3x3());
        Matrix3x3 I_mu_lambda = (this->mu + this->lambda) * Matrix::Identity3x3();
        for (unsigned int i = 0; i <= 2; ++i)
            H[i * 4] = I_mu_lambda; // 0, 4, 8

    }
};


struct M_StVK : public FEM_ContinuousMaterial {
    M_StVK(const scalar _young, const scalar _poisson) : FEM_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensor(const Matrix3x3& F, Matrix3x3& P) override {
        const auto E = getStrainTensor(F);
        const auto trace = Matrix::Trace(E);
        P = this->lambda * trace * F + 2.f * this->mu * F * E;
    }

    virtual scalar getEnergy(const Matrix3x3& F) override {
        const auto E = getStrainTensor(F);
        const auto trace = Matrix::Trace(E);
        // Psi(F) = (lambda / 2) tr(E)² + mu tr(E^2)
        return 0.5f * this->lambda * trace * trace + this->mu * Matrix::SquaredNorm(E);
    }
};


struct M_NeoHooke : public FEM_ContinuousMaterial {
    scalar alpha;
    M_NeoHooke(const scalar _young, const scalar _poisson) : FEM_ContinuousMaterial(_young, _poisson) {
        alpha = 1 + this->mu / this->lambda;
    }

    virtual void getStressTensor(const Matrix3x3& F, Matrix3x3& P) {
        scalar I_3 = glm::determinant(F);
        Matrix3x3 d_detF;
        d_detF[0] = glm::cross(F[1], F[2]);
        d_detF[1] = glm::cross(F[2], F[0]);
        d_detF[2] = glm::cross(F[0], F[1]);
        P = this->lambda * (I_3 - alpha) * d_detF;
    }

    virtual scalar getEnergy(const Matrix3x3& F) {
        scalar I_3 = glm::determinant(F);
        scalar I_2 = Matrix::SquaredNorm(F);
        return 0.5f * this->mu * (I_2 - 3.f) + 0.5f * this->lambda * (I_3 - alpha) * (I_3 - alpha);
    }
};

FEM_ContinuousMaterial* get_fem_material(Material material, scalar young, scalar poisson) {
    switch (material)
    {
    case Hooke: return new M_Hooke(young, poisson);
    case StVK: return new M_StVK(young, poisson);
    case Neo_Hooke: return new M_NeoHooke(young, poisson);
    case Developed_Neohooke: return new M_NeoHooke(young, poisson);
    default:
        std::cout << "Material not found" << std::endl;
        return nullptr;
    }
}