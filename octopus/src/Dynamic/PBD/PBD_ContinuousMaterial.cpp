#include "Dynamic/PBD/PBD_ContinuousMaterial.h"
#include <vector>
#include <iostream>

// PBD_ContinuousMaterial
scalar PBD_ContinuousMaterial::get_energy(const Matrix3x3 &F) {
    Matrix3x3 P;
    scalar energy;
    get_pk1_and_energy(F, P, energy);
    return energy * get_stiffness();
}

Matrix3x3 PBD_ContinuousMaterial::get_pk1(const Matrix3x3 &F) {
    Matrix3x3 P;
    scalar energy;
    get_pk1_and_energy(F, P, energy);
    return P * get_stiffness();
}


// Hooke
void Hooke_First::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    const auto trace = Matrix::Trace(get_strain_linear(F));
    // P(F) = 2E   C(F) = tr(E)�
    energy = trace * trace;
    P = 2.f * trace * Matrix::Identity3x3();
}

void Hooke_Second::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    const auto E = get_strain_linear(F);
    // P(F) = 2E
    // C(F) = tr(E�)
    P = 2.f * E;
    energy = Matrix::SquaredTrace(E);
}


// StVK
void StVK_First::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    const auto trace = Matrix::Trace(get_strain_tensor(F));
    // P(F) = 2 tr(E) F
    P = 2.f * trace * F;

    // C(F) = tr(E)^2
    energy = trace * trace;
}

void StVK_Second::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    const auto E = get_strain_tensor(F);
    // P(F) = 2 F E
    P = 4.f * F * E;
    // C(F) = tr(E^2)
    energy = 2.f * Matrix::SquaredTrace(E);
}


// Neohooke
void VolumePreservation::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    // C(F) = (det(F) - alpha)�
    // P(F) = 2 det(F) det(F)/dx
    scalar I_3 = glm::determinant(F);
    scalar detF = I_3 - alpha;
    scalar shift = this->mu / this->lambda;
    energy = (detF) * (detF);
    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * detF * d_detF;
}

void Stable_NeoHooke_Second::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    energy = Matrix::SquaredNorm(F) - 3.f;
    P = 2.f * F;
}

void NeoHooke_ln_First::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    Matrix3x3 F_t_inv = glm::transpose(glm::inverse(F));
    scalar J = glm::determinant(F);
    P = (2.f * J) * (F - F_t_inv) * F_t_inv;
}


void NeoHooke_ln_Second::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    energy = Matrix::SquaredNorm(F) - 3.f;
    P = 2.f * F;
}

void Developed_Stable_NeoHooke_Second::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    scalar I_3 = glm::determinant(F);
    energy = Matrix::SquaredNorm(F) - scalar(3) - scalar(2) * (I_3 - scalar(1));

    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * F - 2.f * d_detF;
}

// Anysostropic
void Anisotropic::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    Vector3 Fa = F * a;
    scalar IVc_1 = glm::dot(Fa, Fa) - scalar(1);

    energy = (IVc_1) * (IVc_1);

    P = 4.f * IVc_1 * glm::outerProduct(Fa, a);
}

void Sqrt_Anisotropic::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    Vector3 Fa = F * a;
    scalar dFa = glm::length(Fa);

    energy = (dFa - 1.f) * (dFa - 1.f);

    P = 2.f * (1.f - 1.f / dFa) * glm::outerProduct(Fa, a);
}


// Special
// Combine multiple energies that have the same stifness factor
void Material_Union::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    Matrix3x3 temp_P;
    scalar temp_E;
    energy = 0;
    P = Matrix::Zero3x3();
    for (PBD_ContinuousMaterial *m: materials) {
        m->get_pk1_and_energy(F, temp_P, temp_E);
        energy += temp_E;
        P += temp_P;
    }
}


//Muller and Macklin neohooke energy for Tetra
void C_Stable_NeoHooke_First::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    // C(F) = (det(F) - alpha)
    // P(F) = 2 det(F) det(F)/dx
    energy = glm::determinant(F) - alpha;
    P[0] = glm::cross(F[1], F[2]);
    P[1] = glm::cross(F[2], F[0]);
    P[2] = glm::cross(F[0], F[1]);
}


void C_Stable_NeoHooke_Second::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    energy = sqrt(abs(Matrix::SquaredNorm(F) - 3));
    P = F * (scalar(1) / energy);
}


void C_Developed_Stable_NeoHooke_First::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    // C(F) = (det(F) - 1)
    // P(F) = det(F)/dx
    energy = glm::determinant(F) - 1.f;
    P[0] = glm::cross(F[1], F[2]);
    P[1] = glm::cross(F[2], F[0]);
    P[2] = glm::cross(F[0], F[1]);
}

void C_Developed_Stable_NeoHooke_Second::get_pk1_and_energy(const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    energy = Matrix::SquaredNorm(F) - 3.f - 2.f * (glm::determinant(F) - 1.f);
    energy = sqrt(abs(energy));

    Matrix3x3 d_det;
    d_det[0] = glm::cross(F[1], F[2]);
    d_det[1] = glm::cross(F[2], F[0]);
    d_det[2] = glm::cross(F[0], F[1]);
    P = (2.f * F - 2.f * d_det) * (1.f / (2.f * energy));
}

std::vector<PBD_ContinuousMaterial *> get_pbd_materials(Material material, scalar young, scalar poisson) {
    std::vector<PBD_ContinuousMaterial *> materials;
    switch (material) {
        case Hooke:
            materials.push_back(new Hooke_First(young, poisson));
            materials.push_back(new Hooke_Second(young, poisson));
            break;
        case StVK:
            materials.push_back(new StVK_First(young, poisson));
            materials.push_back(new StVK_Second(young, poisson));
            break;
        case NeoHooke:
            materials.push_back(new Stable_NeoHooke_First(young, poisson));
            materials.push_back(new Stable_NeoHooke_Second(young, poisson));
            break;
        case Stable_NeoHooke:
            materials.push_back(new Developed_Stable_NeoHooke_First(young, poisson));
            materials.push_back(new Developed_Stable_NeoHooke_Second(young, poisson));
            break;
        default:
            std::cout << "Material not found" << std::endl;
            break;
    }
    return materials;
}
