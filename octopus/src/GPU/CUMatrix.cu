#include "GPU/PBD/GPU_PBD.h"

#include <Manager/Debug.h>
#include <Manager/Key.h>

__device__ void vec_reduction(const int tid, const int block_size, const int offset, const int v_size, scalar* s_data) {
    __syncthreads();
    int i,b;
    for(i=block_size/2, b=(block_size+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < v_size; ++j) {
                s_data[(offset + tid)*v_size+j] += s_data[(offset + tid+b)*v_size+j];
            }
        }
        __syncthreads();
        i = (b>i) ? b : i;
    }
}

__device__ Matrix3x3 vec_hat(const Vector3 &v) {
    return {
        0.f, -v.z, v.y,
        v.z, 0.f, -v.x,
        -v.y, v.x, 0.f
    };
}

__device__ Matrix3x3 mat3x3_com(const Matrix3x3 &m) {
    return {glm::cross(m[1], m[2]), glm::cross(m[2], m[0]), glm::cross(m[0], m[1])};
}

// global device function
__device__ scalar mat3x3_trace(const Matrix3x3 &m) {
    return m[0][0] + m[1][1] + m[2][2];
}

__device__ scalar squared_trace(const Matrix3x3 &m)
{
    return m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2] + (m[0][1] * m[1][0] + m[1][2] * m[2][1] + m[2][0] * m[0][2]) * 2.f;
}

__device__ scalar squared_norm(const Matrix3x3& m)
{
    scalar st = 0.f;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            st += m[i][j] * m[i][j];
    return st;
}

__device__ bool mat_3x3_is_symmetric(const glm::mat3& A, const scalar eps) {
    return (
        abs(A[0][1] - A[1][0]) < eps &&
        abs(A[0][2] - A[2][0]) < eps &&
        abs(A[1][2] - A[2][1]) < eps
    );
}

__device__ bool mat_3x3_is_positive_definite(const glm::mat3& A) {
    // Mineur d’ordre 1
    float m1 = A[0][0];
    if (m1 <= 0.0f) return false;

    // Mineur d’ordre 2
    float m2 = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (m2 <= 0.0f) return false;

    // Déterminant (mineur d’ordre 3)
    float detA = glm::determinant(A);
    if (detA <= 0.0f) return false;

    return true;
}

__device__ void mat3x3_polardecomposition(const Matrix3x3 &F, Matrix3x3 &R, Matrix3x3 &S, const int max_it, const scalar tol) {
    glm::quat q(R);
    for (unsigned int iter = 0; iter < max_it; iter++) {
        // Calcul du vecteur omega
        const scalar denom = 1.f / fabs(glm::dot(glm::vec3(R[0]), glm::vec3(F[0])) +
                                   glm::dot(glm::vec3(R[1]), glm::vec3(F[1])) +
                                   glm::dot(glm::vec3(R[2]), glm::vec3(F[2]))) + 1.0e-9f;

        const glm::vec3 omega = (glm::cross(glm::vec3(R[0]), glm::vec3(F[0])) +
                          glm::cross(glm::vec3(R[1]), glm::vec3(F[1])) +
                          glm::cross(glm::vec3(R[2]), glm::vec3(F[2]))) * denom;

        const scalar w = glm::length(omega);
        if (w < tol) break;

        glm::quat dq = glm::angleAxis(w, omega / w);

        q = dq * q;
        q = glm::normalize(q);
        R = glm::toMat3(q);
    }
    S = glm::transpose(R) * F;
}

__device__ void print_vec(const Vector3 &v) {
    printf("x:%f y:%f z:%f \n", v.x, v.y, v.z);
}

__device__ void print_mat(const Matrix3x3 &m) {
    printf("|%.3e  %.3e  %.3e|\n|%.3e  %.3e  %.3e|\n|%.3e  %.3e  %.3e|\n\n", m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2],
           m[2][2]);
}

__device__ void print_mat(const Matrix2x2 &m) {
    printf("|%.3e  %.3e|\n|%.3e  %.3e|\n\n", m[0][0], m[1][0],m[0][1], m[1][1]);
}
