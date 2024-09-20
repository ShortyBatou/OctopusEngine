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
void M_Hooke::get_sub_hessian(const Matrix3x3 &, std::vector<Matrix3x3>& H) {
    H.resize(9, Matrix::Zero3x3());
    const Matrix3x3 I_mu_lambda = (this->mu + this->lambda) * Matrix::Identity3x3();
    for (unsigned int i = 0; i <= 2; ++i)
        H[i * 4] = I_mu_lambda; // 0, 4, 8
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
    Matrix3x3 d_detF = Matrix::Com(F);
    return this->lambda * (I_3 - 1.f) * d_detF + this->mu * F;
}

scalar M_NeoHooke::get_energy(const Matrix3x3 &F) {
    const scalar I_3 = glm::determinant(F);
    const scalar I_2 = Matrix::SquaredNorm(F);
    return 0.5f * this->mu * (I_2 - 3.f) + 0.5f * this->lambda * (I_3 - 1.f) * (I_3 - 1.f);
}


Matrix3x3 M_Stable_NeoHooke::get_pk1(const Matrix3x3 &F) {
    const scalar I_3 = glm::determinant(F);
    Matrix3x3 d_detF = Matrix::Com(F);
    return this->lambda * (I_3 - alpha) * d_detF + this->mu * F;
}

scalar M_Stable_NeoHooke::get_energy(const Matrix3x3 &F) {
    const scalar I_3 = glm::determinant(F);
    const scalar I_2 = Matrix::SquaredNorm(F);
    return 0.5f * this->mu * (I_2 - 3.f) + 0.5f * this->lambda * (I_3 - alpha) * (I_3 - alpha);
}

void M_Stable_NeoHooke::get_sub_hessian(const Matrix3x3& F, std::vector<Matrix3x3>& d2W_dF2)
{
    d2W_dF2.resize(9);
    Matrix3x3 comF = Matrix::Com(F);
    scalar detF = glm::determinant(F);
    scalar s = lambda * (detF - alpha);
    // lambda * (I3 - alpha) * H3
    d2W_dF2[0] = Matrix3x3(0);
    d2W_dF2[1] = Matrix::Hat(F[2]) * s;
    d2W_dF2[2] = -Matrix::Hat(F[1]) * s;
    d2W_dF2[3] = -d2W_dF2[1];
    d2W_dF2[4] = Matrix3x3(0);
    d2W_dF2[5] = Matrix::Hat(F[0]) * s;
    d2W_dF2[6] = -d2W_dF2[2];
    d2W_dF2[7] = -d2W_dF2[5];
    d2W_dF2[8] = Matrix3x3(0);

    // mu/2 * H2 = mu * I_9x9x
    for (int i = 0; i < 3; ++i) {
        d2W_dF2[0][i][i] += mu;
        d2W_dF2[4][i][i] += mu;
        d2W_dF2[8][i][i] += mu;
    }

    // lambda vec(com F) * vec(com F)^T
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            d2W_dF2[i*3 + j] += glm::outerProduct(comF[i], comF[j]) * lambda;

}


FEM_ContinuousMaterial *get_fem_material(const Material material, const scalar young, const scalar poisson) {
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
