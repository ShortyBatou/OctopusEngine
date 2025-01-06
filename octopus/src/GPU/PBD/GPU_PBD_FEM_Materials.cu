#include "GPU/PBD/GPU_PBD_FEM.h"
#include <Manager/Debug.h>
#include <GPU/CUMatrix.h>

__device__ void stvk_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const scalar trace = mat3x3_trace(0.5f * (glm::transpose(F) * F - Matrix3x3(1.f)));
    C = trace * trace;
    P = (2.f * trace * F);
}

__device__ void stvk_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const Matrix3x3 E = 0.5f * (glm::transpose(F) * F - Matrix3x3(1.f));
    P = 4.f * F * E;
    C = 2.f * squared_trace(E);
}

__device__ void hooke_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const float trace = mat3x3_trace(0.5f * (glm::transpose(F) + F) - Matrix3x3(1.f));
    // P(F) = 2E   C(F) = tr(E)�
    C = trace * trace;
    P = 2.f * trace * Matrix3x3(1.f);
}

__device__ void hooke_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    const Matrix3x3 E = 0.5f * (glm::transpose(F) + F) - Matrix3x3(1.f);
    // P(F) = 2E   C(F) = tr(E)�
    C = 2.f * squared_trace(E);
    P = 4.f * E;
}


__device__ void snh_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C, scalar alpha) {
    const scalar I_3 = glm::determinant(F);
    const scalar detF = I_3 - alpha;
    C = (detF) * (detF);
    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * detF * d_detF;
}

__device__ void snh_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    C = squared_norm(F) - 3.f;
    P = 2.f * F;
}

__device__ void dsnh_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    scalar I_3 = glm::determinant(F);
    scalar detF = I_3 - 1;
    C = (detF) * (detF);
    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * detF * d_detF;
}

__device__ void dsnh_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C) {
    scalar I_3 = glm::determinant(F);
    C = squared_norm(F) - 3.f - 2.f * (I_3 - 1.f);
    Matrix3x3 d_detF; // derivative of det(F) by F
    d_detF[0] = glm::cross(F[1], F[2]);
    d_detF[1] = glm::cross(F[2], F[0]);
    d_detF[2] = glm::cross(F[0], F[1]);
    P = 2.f * F - 2.f * d_detF;
}

__device__ void eval_material(const Material material, const int m, const scalar lambda, const scalar mu, const Matrix3x3 &F, Matrix3x3 &P, scalar &energy) {
    switch (material) {
        case Hooke :
            if (m == 0) hooke_first(F, P, energy);
            else hooke_second(F, P, energy); break;
        case StVK :
            if (m == 0) stvk_first(F, P, energy);
            else stvk_second(F, P, energy); break;
        case NeoHooke :
            if (m == 0) snh_first(F, P, energy, 1.f + mu / lambda);
            else snh_second(F, P, energy); break;
        case Stable_NeoHooke :
            if (m == 0) dsnh_first(F, P, energy);
            else dsnh_second(F, P, energy); break;
    }
}