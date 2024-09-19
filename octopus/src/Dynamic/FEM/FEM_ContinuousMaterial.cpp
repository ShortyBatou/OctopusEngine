 #include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include <vector>
#include <iostream>

Matrix3x3 M_Hooke::get_pk1(const Matrix3x3 &F) {
    const auto E = get_strain_linear(F);
    return this->lambda * Matrix::Trace(E) * Matrix::Identity3x3() + this->mu * E;
}

scalar M_Hooke::get_energy(const Matrix3x3 &F) {
    const auto E = get_strain_linear(F);
    const auto trace = Matrix::Trace(E);
    // Psi(F) = lambda / 2 tr(E)� + mu tr(E^2)
    return 0.5f * this->lambda * trace * trace + this->mu * Matrix::SquaredNorm(E);
}

// get all H_kl of dF_i**T H_kl dF_j
std::vector<Matrix3x3> M_Hooke::get_sub_hessian(const Matrix3x3 &) {
    std::vector H(9, Matrix::Zero3x3());
    const Matrix3x3 I_mu_lambda = (this->mu + this->lambda) * Matrix::Identity3x3();
    for (unsigned int i = 0; i <= 2; ++i)
        H[i * 4] = I_mu_lambda; // 0, 4, 8
    return H;
}

Matrix3x3 M_StVK::get_pk1(const Matrix3x3 &F) {
    const auto E = get_strain_tensor(F);
    const auto trace = Matrix::Trace(E);
    return this->lambda * trace * F + 2.f * this->mu * F * E;
}

scalar M_StVK::get_energy(const Matrix3x3 &F) {
    const auto E = get_strain_tensor(F);
    const auto trace = Matrix::Trace(E);
    // Psi(F) = (lambda / 2) tr(E)� + mu tr(E^2)
    return 0.5f * this->lambda * trace * trace + this->mu * Matrix::SquaredNorm(E);
}


Matrix3x3 M_NeoHooke::get_pk1(const Matrix3x3 &F) {
    scalar I_3 = glm::determinant(F);
    Matrix3x3 d_detF;
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    return this->lambda * (I_3 - 1.f) * d_detF + this->mu * F;
}

scalar M_NeoHooke::get_energy(const Matrix3x3 &F) {
    const scalar I_3 = glm::determinant(F);
    const scalar I_2 = Matrix::SquaredNorm(F);
    return 0.5f * this->mu * (I_2 - 3.f) + 0.5f * this->lambda * (I_3 - 1.f) * (I_3 - 1.f);
}


Matrix3x3 M_Stable_NeoHooke::get_pk1(const Matrix3x3 &F) {
    const scalar I_3 = glm::determinant(F);
    Matrix3x3 d_detF;
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    return this->lambda * (I_3 - alpha) * d_detF + this->mu * F;
}

scalar M_Stable_NeoHooke::get_energy(const Matrix3x3 &F) {
    const scalar I_3 = glm::determinant(F);
    const scalar I_2 = Matrix::SquaredNorm(F);
    return 0.5f * this->mu * (I_2 - 3.f) + 0.5f * this->lambda * (I_3 - alpha) * (I_3 - alpha);
}


FEM_ContinuousMaterial *get_fem_material(Material material, scalar young, scalar poisson) {
    switch (material) {
        case Hooke: return new M_Hooke(young, poisson);
        case StVK: return new M_StVK(young, poisson);
        case NeoHooke: return new M_NeoHooke(young, poisson);
        case Stable_NeoHooke: return new M_Stable_NeoHooke(young, poisson);
        default:
            std::cout << "Material not found" << std::endl;
            return nullptr;
    }
}
