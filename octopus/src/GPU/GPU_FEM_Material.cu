#include "GPU/GPU_FEM.h"
#include <GPU/CUMatrix.h>
#include "GPU/GPU_FEM_Material.h"

__device__ Matrix3x3 get_strain_linear(const Matrix3x3& F) {
    return 0.5f * (glm::transpose(F) + F) - Matrix3x3(1.f);

}

__device__ Matrix3x3 get_strain(const Matrix3x3& F) {
    return 0.5f * (glm::transpose(F) * F - Matrix3x3(1.f));
}

__device__ Matrix3x3 hooke_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu) {
    const Matrix3x3 E = get_strain_linear(F);
    return  Matrix3x3(lambda * mat3x3_trace(E)) + mu * E;
}

__device__ Matrix3x3 stvk_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu) {
    const Matrix3x3 E = get_strain(F);
    const scalar trace = mat3x3_trace(E);
    return lambda * trace * F + 2.f * mu * F * E;
}

__device__ Matrix3x3 nh_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu) {
    const scalar I_3 = glm::determinant(F);
    const Matrix3x3 d_detF = mat3x3_com(F);
    return lambda * (I_3 - 1.f) * d_detF + mu * F;
}

__device__ Matrix3x3 snh_stress(const Matrix3x3 &F, const scalar lambda, const scalar mu) {
    const scalar I_3 = glm::determinant(F);
    const Matrix3x3 d_detF = mat3x3_com(F);
    const scalar alpha = 1.f + (mu / lambda);
    return lambda * (I_3 - alpha) * d_detF + mu * F;
}


__device__ void hooke_hessian(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 d2W_dF2[9]) {
    for(int i = 0; i < 9; ++i)
        d2W_dF2[i] = Matrix3x3(0.f);

    const Matrix3x3 I_mu_lambda = Matrix3x3(mu + lambda);
    for (int i = 0; i <= 2; ++i)
        d2W_dF2[i * 4] = I_mu_lambda; // 0, 4, 8
}

__device__ void stvk_hessian(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 d2W_dF2[9]) {
    for(int i = 0; i < 9; ++i)
        d2W_dF2[i] = Matrix3x3(0.f);

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

__device__ void nh_hessian(const Matrix3x3 &F, const scalar lambda, const scalar mu, Matrix3x3 d2W_dF2[9]) {
    Matrix3x3 comF = mat3x3_com(F);
    const scalar detF = glm::determinant(F);
    const scalar s = lambda * (detF - 1);
    // lambda * (I3 - 1) * H3
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


__device__ void hooke_hessian_sym(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 d2W_dF2[6]) {
    for(int i = 0; i <= 5; ++i)
        d2W_dF2[i] = Matrix3x3(0.f);

    const Matrix3x3 I_mu_lambda = Matrix3x3(mu + lambda);
    d2W_dF2[0] = I_mu_lambda;
    d2W_dF2[3] = I_mu_lambda;
    d2W_dF2[5] = I_mu_lambda;
}

__device__ void stvk_hessian_sym(const Matrix3x3 &F, const scalar lambda, const scalar mu,Matrix3x3 d2W_dF2[6]) {
    const Matrix3x3 FFt = F * glm::transpose(F);
    const Matrix3x3 FtF = glm::transpose(F) * F;
    const Matrix3x3 H1 = Matrix3x3(0.5f * lambda * mat3x3_trace(FFt) - mu);
    const Matrix3x3 H2_A = mu * FFt;
    const Matrix3x3 diag = H1 + H2_A;

    d2W_dF2[1] = Matrix3x3(0);
    d2W_dF2[2] = Matrix3x3(0);
    d2W_dF2[4] = Matrix3x3(0);

    d2W_dF2[0] = diag;
    d2W_dF2[3] = diag;
    d2W_dF2[5] = diag;

    for (int i = 0, k = 0; i <= 2; ++i) {
        for (int j = i; j <= 2; ++j, ++k) {
            d2W_dF2[k] += 0.5f * lambda * glm::outerProduct(F[i], F[j]);
            d2W_dF2[k] += mu * glm::outerProduct(F[j], F[i]);
            d2W_dF2[k] += Matrix3x3(FtF[i][j]);
        }
    }
}

__device__ void nh_hessian_sym(const Matrix3x3 &F, const scalar lambda, const scalar mu, Matrix3x3 d2W_dF2[6]) {
    const Matrix3x3 comF = mat3x3_com(F);
    const scalar detF = glm::determinant(F);
    const scalar s = lambda * (detF - 1.);
    // lambda * (I3 - 1) * H3
    d2W_dF2[1] = vec_hat(F[2]) * s;
    d2W_dF2[2] = -vec_hat(F[1]) * s;
    d2W_dF2[4] = vec_hat(F[0]) * s;

    // mu/2 * H2 = mu * I_9x9x
    d2W_dF2[0] = Matrix3x3(mu);
    d2W_dF2[3] = Matrix3x3(mu);
    d2W_dF2[5] = Matrix3x3(mu);

    // lambda vec(com F) * vec(com F)^T
    for (int i = 0, k = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j, ++k)
            d2W_dF2[k] += glm::outerProduct(comF[j], comF[i]) * lambda;
}


__device__ void snh_hessian_sym(const Matrix3x3 &F, const scalar lambda, const scalar mu, Matrix3x3 d2W_dF2[6]) {
    const Matrix3x3 comF = mat3x3_com(F);
    const scalar detF = glm::determinant(F);
    const scalar alpha = 1.f + (mu / lambda  );
    const scalar s = lambda * (detF - alpha);
    // lambda * (I3 - alpha) * H3
    d2W_dF2[1] = vec_hat(F[2]) * s;
    d2W_dF2[2] = -vec_hat(F[1]) * s;
    d2W_dF2[4] = vec_hat(F[0]) * s;

    // mu/2 * H2 = mu * I_9x9x
    d2W_dF2[0] = Matrix3x3(mu);
    d2W_dF2[3] = Matrix3x3(mu);
    d2W_dF2[5] = Matrix3x3(mu);

    // lambda vec(com F) * vec(com F)^T
    for (int i = 0, k = 0; i < 3; ++i)
        for (int j = i; j < 3; ++j, ++k)
            d2W_dF2[k] += glm::outerProduct(comF[j], comF[i]) * lambda;
}

__device__ Matrix3x3 eval_pk1_stress(const Material material, const scalar lambda, const scalar mu, const Matrix3x3 &F) {
    switch (material) {
        case Hooke : return hooke_stress(F, lambda, mu);
        case StVK : return stvk_stress(F, lambda, mu);
        case NeoHooke : return nh_stress(F, lambda, mu);
        case Stable_NeoHooke : return snh_stress(F, lambda, mu);
    }
    return Matrix3x3(0.f);
}

__device__ Matrix3x3 pk1_to_cauchy_stress(const Matrix3x3 &F, const Matrix3x3 &P) {
    return P * glm::transpose(F) * (1.f / glm::determinant(F));
}

__device__ scalar von_mises_stress(const Matrix3x3 &C) {
    const scalar s0_1 = C[0][0] - C[1][1];
    const scalar s1_2 = C[1][1] - C[2][2];
    const scalar s0_2 = C[0][0] - C[2][2];
    scalar s = 0.5f * (s0_1 * s0_1 + s1_2 * s1_2 + s0_2 * s0_2);
    s += 3.f * (C[0][1] * C[0][1] + C[1][2] * C[1][2] + C[0][2] * C[0][2]);
    return sqrt(s);
}

__device__ void eval_hessian(const Material material, const scalar lambda, const scalar mu, const Matrix3x3 &F, Matrix3x3 d2W_dF2[9]) {
    switch (material) {
        case Hooke : hooke_hessian(F, lambda, mu, d2W_dF2); break;
        case StVK : stvk_hessian(F, lambda, mu, d2W_dF2); break;
        case NeoHooke : nh_hessian(F, lambda, mu, d2W_dF2); break;
        case Stable_NeoHooke : snh_hessian(F, lambda, mu, d2W_dF2); break;
    }
}

__device__ void eval_hessian_sym(const Material material, const scalar lambda, const scalar mu, const Matrix3x3 &F, Matrix3x3 d2W_dF2[6]) {
    switch (material) {
        case Hooke : hooke_hessian_sym(F, lambda, mu, d2W_dF2); break;
        case StVK : stvk_hessian_sym(F, lambda, mu, d2W_dF2); break;
        case NeoHooke : snh_hessian_sym(F, lambda, mu, d2W_dF2); break;
        case Stable_NeoHooke : snh_hessian_sym(F, lambda, mu, d2W_dF2); break;
    }
}

__device__ Matrix3x3 assemble_sub_hessian_sym(const Vector3& dF_dx, const scalar& V, Matrix3x3 d2W_dF2[6])
{
    Matrix3x3 H(0);
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
        H += dF_dx[i] * dF_dx[j] * d2W_dF2[max(i,j)][min(i,j)];
    }}
    return H * V;
}

__device__ Matrix3x3 assemble_sub_hessian(const Vector3& dF_dx, const scalar& V, Matrix3x3 d2W_dF2[9])
{
    Matrix3x3 H(0);
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            H += dF_dx[i] * dF_dx[j] * d2W_dF2[i][j];
        }}
    return H * V;
}