#pragma once
#include "Core/Base.h"

#define GAMMA 5.8284271247
#define C_STAR 0.9238795325
#define S_STAR 0.3826834323
#define SVD_EPS 0.0000001

//https://gist.github.com/alexsr/5065f0189a7af13b2f3bc43d22aff62f
namespace MatrixAlgo {
    Vector2 approx_givens_quat(float s_pp, float s_pq, float s_qq) {
        float c_h = 2 * (s_pp - s_qq);
        float s_h2 = s_pq * s_pq;
        float c_h2 = c_h * c_h;
        if (GAMMA * s_h2 < c_h2) {
            float omega = 1.0f / sqrt(s_h2 + c_h2);
            return Vector2(omega * c_h, omega * s_pq);
        }
        return Vector2(C_STAR, S_STAR);
    }

    // the quaternion is stored in vec4 like so:
    // (c, s * vec3) meaning that .x = c
    void quat_to_mat3(const Vector4& quat, Matrix3x3& mat) {
        const float qx2 = quat.y * quat.y;
        const float qy2 = quat.z * quat.z;
        const float qz2 = quat.w * quat.w;
        const float qwqx = quat.x * quat.y;
        const float qwqy = quat.x * quat.z;
        const float qwqz = quat.x * quat.w;
        const float qxqy = quat.y * quat.z;
        const float qxqz = quat.y * quat.w;
        const float qyqz = quat.z * quat.w;

        mat = Matrix3x3(
            1.0f - 2.0f * (qy2 + qz2), 2.0f * (qxqy + qwqz), 2.0f * (qxqz - qwqy),
            2.0f * (qxqy - qwqz), 1.0f - 2.0f * (qx2 + qz2), 2.0f * (qyqz + qwqx),
            2.0f * (qxqz + qwqy), 2.0f * (qyqz - qwqx), 1.0f - 2.0f * (qx2 + qy2));
    }

    Matrix3x3 symmetric_eigenanalysis(const Matrix3x3& A) {
        Matrix3x3 S = glm::transpose(A) * A;
        // jacobi iteration
        Matrix3x3 q = Matrix3x3(1.0f);
        for (int i = 0; i < 20; i++) {
            Vector2 ch_sh = approx_givens_quat(S[0].x, S[0].y, S[1].y);
            Vector4 ch_sh_quat = Vector4(ch_sh.x, 0, 0, ch_sh.y);
            Matrix3x3 q_mat; quat_to_mat3(ch_sh_quat, q_mat);
            S = glm::transpose(q_mat) * S * q_mat;
            q = q * q_mat;

            ch_sh = approx_givens_quat(S[0].x, S[0].z, S[2].z);
            ch_sh_quat = Vector4(ch_sh.x, 0, -ch_sh.y, 0);
            quat_to_mat3(ch_sh_quat, q_mat);
            S = glm::transpose(q_mat) * S * q_mat;
            q = q * q_mat;

            ch_sh = approx_givens_quat(S[1].y, S[1].z, S[2].z);
            ch_sh_quat = Vector4(ch_sh.x, ch_sh.y, 0, 0);
            quat_to_mat3(ch_sh_quat, q_mat);
            S = glm::transpose(q_mat) * S * q_mat;
            q = q * q_mat;
        }
        return q;
    }

    Vector2 approx_qr_givens_quat(float a0, float a1) {
        float rho = sqrt(a0 * a0 + a1 * a1);
        float s_h = a1;
        float max_rho_eps = rho;
        if (rho <= SVD_EPS) {
            s_h = 0;
            max_rho_eps = SVD_EPS;
        }
        float c_h = max_rho_eps + a0;
        if (a0 < 0) {
            float temp = c_h - 2 * a0;
            c_h = s_h;
            s_h = temp;
        }
        float omega = 1.0f / sqrt(c_h * c_h + s_h * s_h);
        return  Vector2(omega * c_h, omega * s_h);
    }

    void qr_decomp(const Matrix3x3& B, Matrix3x3& Q, Matrix3x3& R) {
        // 1 0
        // (ch, 0, 0, sh)
        Matrix3x3 Q10, Q20, Q21;

        Vector2 ch_sh10 = approx_qr_givens_quat(B[0].x, B[0].y);
        quat_to_mat3(Vector4(ch_sh10.x, 0, 0, ch_sh10.y), Q10);
        R = glm::transpose(Q10) * B;

        // 2 0
        // (ch, 0, -sh, 0)
        Vector2 ch_sh20 = approx_qr_givens_quat(R[0].x, R[0].z);
        quat_to_mat3(Vector4(ch_sh20.x, 0, -ch_sh20.y, 0), Q20);
        R = glm::transpose(Q20) * R;

        // 2 1
        // (ch, sh, 0, 0)
        Vector2 ch_sh21 = approx_qr_givens_quat(R[1].y, R[1].z);
        quat_to_mat3(Vector4(ch_sh21.x, ch_sh21.y, 0, 0), Q21);
        R = glm::transpose(Q21) * R;

        Q = Q10 * Q20 * Q21;
    }

    void SVD(const Matrix3x3& A, Matrix3x3& U, Matrix3x3& S, Matrix3x3& V) {
        V = symmetric_eigenanalysis(A);

        Matrix3x3 B = A * V;

        // sort singular values
        float rho0 = dot(B[0], B[0]);
        float rho1 = dot(B[1], B[1]);
        float rho2 = dot(B[2], B[2]);
        if (rho0 < rho1) {
            B[0] *= -1;
            V[0] *= -1;

            std::swap(B[0], B[1]);
            std::swap(V[0], V[1]);
            std::swap(rho0, rho1);
        }
        if (rho0 < rho2) {
            B[0] *= -1;
            V[0] *= -1;

            std::swap(B[0], B[2]);
            std::swap(V[0], V[2]);

            rho2 = rho0;
        }
        if (rho1 < rho2) {
            B[1] *= -1;
            V[1] *= -1;

            std::swap(B[1], B[2]);
            std::swap(V[1], V[2]);
        }

        qr_decomp(B, U, S);
    }


    void SVD_To_Polar(
        const Matrix3x3& U, const Matrix3x3& S, const Matrix3x3& V, 
        Matrix3x3& Up, Matrix3x3& P) {
        P = V * S * glm::transpose(V);
        Up = U * glm::transpose(V);
    }

    void Polar_Decomp(const Matrix3x3& A, Matrix3x3& U, Matrix3x3& P) {
        Matrix3x3 S, V;
        SVD(A, U, S, V);
        P = V * S * glm::transpose(V);
        U = U * transpose(V);
    }
};