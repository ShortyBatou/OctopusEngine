#pragma once
#include "Core/Base.h"

scalar Matrix::Trace(const Matrix2x2 &m) {
    return m[0][0] + m[1][1];
}

scalar Matrix::Trace(const Matrix3x3 &m) {
    return m[0][0] + m[1][1] + m[2][2];
}

scalar Matrix::Trace(const Matrix4x4 &m) {
    return m[0][0] + m[1][1] + m[2][2] + m[3][3];
}

// tr(m�)
scalar Matrix::SquaredTrace(const Matrix2x2 &m) {
    return m[0][0] * m[0][0] + m[1][0] * m[0][1] * 2.0f + m[1][1] * m[1][1];
}

scalar Matrix::SquaredTrace(const Matrix3x3 &m) {
    return (m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2]) +
           (m[0][1] * m[1][0] + m[1][2] * m[2][1] + m[2][0] * m[0][2]) * 2.f;
}

scalar Matrix::SquaredTrace(const Matrix4x4 &m) {
    return (m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2] + m[3][3] * m[3][3]) +
           (m[0][1] * m[1][0] + m[0][2] * m[2][0] + m[0][3] * m[3][0] + m[1][2] * m[2][1] + m[1][3] * m[3][1] + m[2][3]
            * m[3][2]) * 2.f;
}

// tr(m^T m)
scalar Matrix::SquaredNorm(const Matrix2x2 &m) {
    scalar st(0.f);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            st += m[i][j] * m[i][j];
    return st;
}

scalar Matrix::SquaredNorm(const Matrix3x3 &m) {
    scalar st(0.f);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            st += m[i][j] * m[i][j];
    return st;
}

scalar Matrix::SquaredNorm(const Matrix4x4 &m) {
    scalar st(0.f);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            st += m[i][j] * m[i][j];
    return st;
}
