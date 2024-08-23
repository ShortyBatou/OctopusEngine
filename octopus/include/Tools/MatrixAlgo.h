#pragma once
#include "Core/Base.h"

#define GAMMA 5.8284271247f
#define C_STAR 0.9238795325f
#define S_STAR 0.3826834323f
#define SVD_EPS 0.0000001f

//https://gist.github.com/alexsr/5065f0189a7af13b2f3bc43d22aff62f
namespace MatrixAlgo {
    Vector2 approx_givens_quat(float s_pp, float s_pq, float s_qq);

    // the quaternion is stored in vec4 like so:
    // (c, s * vec3) meaning that .x = c
    void quat_to_mat3(const Vector4 &quat, Matrix3x3 &mat);

    Matrix3x3 symmetric_eigenanalysis(const Matrix3x3 &A);

    Vector2 approx_qr_givens_quat(float a0, float a1);

    void qr_decomp(const Matrix3x3 &B, Matrix3x3 &Q, Matrix3x3 &R);

    void SVD(const Matrix3x3 &A, Matrix3x3 &U, Matrix3x3 &S, Matrix3x3 &V);


    void SVD_To_Polar(const Matrix3x3 &U, const Matrix3x3 &S, const Matrix3x3 &V, Matrix3x3 &Up, Matrix3x3 &P);

    void Polar_Decomp(const Matrix3x3 &A, Matrix3x3 &U, Matrix3x3 &P);
};
