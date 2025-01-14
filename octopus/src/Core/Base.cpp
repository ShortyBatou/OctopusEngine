#pragma once
#include "Core/Base.h"
#include <math.h>
#include <algorithm>
#include <iostream>
#include <ostream>

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

Matrix3x3 Matrix::Hat(const Vector3 &v) {
    return {
        0.f, -v.z, v.y,
        v.z, 0.f, -v.x,
        -v.y, v.x, 0.f
    };
}

Matrix3x3 Matrix::Com(const Matrix3x3 &m) {
    return {glm::cross(m[1], m[2]), glm::cross(m[2], m[0]), glm::cross(m[0], m[1])};
}

scalar cubic_max_abs_root(const scalar a, const scalar b, const scalar c) {
    const scalar q = (a * a - 3.0f * b) / 9.0f;
    const scalar r = (2.0f * a * a * a - 9.0f * a * b + 27.0f * c) / 54.0f;

    scalar x;
    if (r * r < q * q * q) {
        const scalar sqrt_q = -2.0f * sqrtf(q);
        // Three Real Roots
        const scalar t = acos(std::clamp(r / sqrt(q * q * q), -1.0f, 1.0f));
        const scalar x0 = sqrt_q * cos((t) / 3.0f) - a / 3.0f;
        const scalar x1 = sqrt_q * cos((t + 2.0f * PI) / 3.0f) - a / 3.0f;
        const scalar x2 = sqrt_q * cos((t - 2.0f * PI) / 3.0f) - a / 3.0f;
        x = abs(x0) > abs(x1) && abs(x0) > abs(x2) ? x0 : abs(x1) > abs(x2) && abs(x1) > abs(x0) ? x1 : x2;
    } else {
        // One Real Root
        const scalar e = pow(sqrt(r * r - q * q * q) + abs(r), 1.0f / 3.0f) * (r > 0.0f ? 1.f : -1.f);
        const scalar f = e == 0.0f ? 0.0f : q / e;
        x = (e + f) - a / 3.0f;
    }
    return x;
}

scalar mat3_f_trace_cg(const scalar A, const scalar B, const scalar C, Vector3 &s) {
    // Compute polynomial coefficients
    const scalar b = -2.0f * A;
    const scalar c = -8.0f * C;
    const scalar d = -A * A + 2.0f * B;

    // Find root with largest magnitude using cubic resolvent coefficients
    const scalar y = cubic_max_abs_root(-b, -4.0f * d, -c * c + 4.0f * b * d);

    // Find quadratics for each pair of quartic roots
    scalar q1, p1, q2, p2;

    const scalar D = y * y - 4.0f * d;
    if (D < 1e-10f) {
        const float D2 = std::max(-4.0f * (b - y), 0.0f);
        q1 = q2 = y * 0.5f;
        p1 = +sqrtf(D2) * 0.5f;
        p2 = -sqrtf(D2) * 0.5f;
    } else {
        q1 = (y + sqrtf(D)) * 0.5f;
        q2 = (y - sqrtf(D)) * 0.5f;
        p1 = (-c) / (q1 - q2);
        p2 = (+c) / (q1 - q2);
    }

    // Find first two roots
    const scalar D01 = std::max(p1 * p1 - 4.0f * q1, 0.0f);
    const scalar x0 = (-p1 + sqrtf(D01)) * 0.5f;
    const scalar x1 = (-p1 - sqrtf(D01)) * 0.5f;

    // Find second two roots
    const scalar D23 = std::max(p2 * p2 - 4.0f * q2, 0.0f);
    const scalar x2 = (-p2 - sqrtf(D23)) * 0.5f;
    const scalar x3 = (-p2 + sqrtf(D23)) * 0.5f;

    // Singular Values
    s.x = (x0 + x3) * 0.5f;
    s.y = (x1 + x3) * 0.5f;
    s.z = (x2 + x3) * 0.5f;

    // return trace root
    return x3;
}

void Matrix::PolarDecomposition(const Matrix3x3 &m, Matrix3x3 &R, Matrix3x3 &S) {
    const scalar a = SquaredNorm(m);
    const scalar b = SquaredNorm(glm::transpose(m) * m);
    const scalar c = glm::determinant(m);
    Vector3 s;
    const scalar f = mat3_f_trace_cg(a, b, c, s);
    scalar denom = 4.0f * f * f * f - 4.0f * a * f - 8.0f * c;
    if (denom < eps) {
        R = Matrix::Identity3x3();
        S = m;
    }

    denom = 1.f / denom;

    const float dfdA = 2.0f * (f * f + a) * denom;
    const float dfdB = -2.0f * denom;
    const float dfdC = 8.0f * f * denom;

    const Matrix3x3 dAdM = 2.0f * m;
    const Matrix3x3 dBdM = 4.0f * m * glm::transpose(m) * m;
    const Matrix3x3 dCdM = Com(m);
    R = dfdA * dAdM + dfdB * dBdM + dfdC * dCdM;
    S = glm::transpose(R) * m;
}

Vector3 Matrix::SingularValues(const Matrix3x3& m) {
    const scalar a = SquaredNorm(m);
    const scalar b = SquaredNorm(glm::transpose(m) * m);
    const scalar c = glm::determinant(m);
    Vector3 s; mat3_f_trace_cg(a, b, c, s);
    return s;
}