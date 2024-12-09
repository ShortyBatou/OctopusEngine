#include "GPU/GPU_FEM.h"
#include <GPU/CUMatrix.h>

__device__ Matrix3x3 get_strain_linear(const Matrix3x3& F) {
    return 0.5f * (glm::transpose(F) + F) - Matrix3x3(1.f);

}

__device__ Matrix3x3 get_strain(const Matrix3x3& F) {
    return 0.5f * (glm::transpose(F) * F - Matrix3x3(1.f));
}

__device__ void hooke_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 &P) {
    const Matrix3x3 E = get_strain_linear(F);
    P =  Matrix3x3(lambda * mat3x3_trace(E)) + mu * E;
}

__device__ void stvk_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 &P) {
    const Matrix3x3 E = get_strain(F);
    const scalar trace = mat3x3_trace(E);
    P = lambda * trace * F + 2.f * mu * F * E;
}

__device__ void snh_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu, Matrix3x3 &P) {
    const scalar I_3 = glm::determinant(F);
    const Matrix3x3 d_detF = mat3x3_com(F);
    const scalar alpha = 1.f + (mu / lambda);
    P = lambda * (I_3 - alpha) * d_detF + mu * F;
}

__device__ void hooke_hessian(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 d2W_dF2[9]) {
    for(int i = 0; i <= 2; ++i)
        for(int j = 0; j <= 2; ++j)
            d2W_dF2[i * 3 + j] = Matrix3x3(0.f);

    const Matrix3x3 I_mu_lambda = Matrix3x3(mu + lambda);
    for (int i = 0; i <= 2; ++i)
        d2W_dF2[i * 4] = I_mu_lambda; // 0, 4, 8
}

__device__ void stvk_hessian(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 d2W_dF2[9]) {
    for(int i = 0; i <= 2; ++i)
        for(int j = 0; j <= 2; ++j)
            d2W_dF2[i * 3 + j] = Matrix3x3(0.f);

    const Matrix3x3 FFt = F * glm::transpose(F);
    const Matrix3x3 FtF = glm::transpose(F) * F;
    const Matrix3x3 H1 = Matrix3x3(0.5f * lambda * mat3x3_trace(FFt) - mu);
    const Matrix3x3 H2_A = mu * FFt;
    const Matrix3x3 diag = H1 + H2_A;

    for (int i = 0; i <= 2; ++i) {
        d2W_dF2[i * 4] += diag; // 0, 4, 8
        for (int j = i; j <= 2; ++j) {
            // 0, 1, 2, 4, 5, 8
            d2W_dF2[i * 3 + j] += 0.5f * lambda * glm::outerProduct(F[i], F[j]);
            d2W_dF2[i * 3 + j] += mu * glm::outerProduct(F[j], F[i]);
            d2W_dF2[i * 3 + j] += Matrix3x3(FtF[i][j]);
        }
    }

    // symmetric matrix
    d2W_dF2[3] = d2W_dF2[1];
    d2W_dF2[6] = d2W_dF2[2];
    d2W_dF2[7] = d2W_dF2[5];
}

__device__ void snh_hessian(const Matrix3x3 &F, const scalar lambda, const scalar mu, Matrix3x3 d2W_dF2[9]) {
    Matrix3x3 comF = mat3x3_com(F);
    const scalar detF = glm::determinant(F);
    const scalar alpha = 1.f + (mu / lambda  );
    const scalar s = lambda * (detF - alpha);
    // lambda * (I3 - alpha) * H3
    d2W_dF2[0] = Matrix3x3(0);
    d2W_dF2[1] = vec_hat(F[2]) * s;
    d2W_dF2[2] = -vec_hat(F[1]) * s;
    d2W_dF2[3] = -d2W_dF2[1];
    d2W_dF2[4] = Matrix3x3(0);
    d2W_dF2[5] = vec_hat(F[0]) * s;
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
            d2W_dF2[i * 3 + j] += glm::outerProduct(comF[i], comF[j]) * lambda;
}

__device__ void eval_stress(const Material material, const scalar lambda, const scalar mu, const Matrix3x3 &F, Matrix3x3 &P) {
    switch (material) {
        case Hooke : hooke_stress(F, lambda, mu, P); break;
        case StVK : stvk_stress(F, lambda, mu, P); break;
        case NeoHooke : snh_stress(F, lambda, mu, P); break;
        case Stable_NeoHooke : snh_stress(F, lambda, mu, P); break;
    }
}

__device__ void eval_hessian(const Material material, const scalar lambda, const scalar mu, const Matrix3x3 &F, Matrix3x3 d2W_dF2[9]) {
    switch (material) {
        case Hooke : hooke_hessian(F, lambda, mu, d2W_dF2); break;
        case StVK : stvk_hessian(F, lambda, mu, d2W_dF2); break;
        case NeoHooke : snh_hessian(F, lambda, mu, d2W_dF2); break;
        case Stable_NeoHooke : snh_hessian(F, lambda, mu, d2W_dF2); break;
    }
}