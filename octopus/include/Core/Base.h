#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using scalar = float;
const scalar min_limit = std::numeric_limits<scalar>::min();
const scalar max_limit = std::numeric_limits<scalar>::max();
const scalar eps = min_limit * scalar(1e5);
const glm::precision precision = glm::precision::defaultp;


typedef glm::vec<1, scalar, precision> Vector1;
typedef glm::vec<2, scalar, precision> Vector2;
typedef glm::vec<3, scalar, precision> Vector3;
typedef glm::vec<4, scalar, precision> Vector4;
typedef glm::vec<4, scalar, precision> Color;

typedef glm::vec<1, int, precision> Vector1I;
typedef glm::vec<2, int, precision> Vector2I;
typedef glm::vec<3, int, precision> Vector3I;
typedef glm::vec<4, int, precision> Vector4I;

typedef glm::vec<1, unsigned int, precision> Vector1UI;
typedef glm::vec<2, unsigned int, precision> Vector2UI;
typedef glm::vec<3, unsigned int, precision> Vector3UI;
typedef glm::vec<4, unsigned int, precision> Vector4UI;

typedef glm::mat<2, 2, scalar, precision> Matrix2x2; 
typedef glm::mat<3, 3, scalar, precision> Matrix3x3; 
typedef glm::mat<4, 4, scalar, precision> Matrix4x4; 

struct Matrix
{
    static Matrix2x2 Zero2x2() { return Matrix2x2(0.); }
    static Matrix3x3 Zero3x3() { return Matrix3x3(0.); }
    static Matrix4x4 Zero4x4() { return Matrix4x4(0.); }

    static Matrix2x2 Identity2x2() { return Matrix2x2(1.); }
    static Matrix3x3 Identity3x3() { return Matrix3x3(1.); }
    static Matrix4x4 Identity4x4() { return Matrix4x4(1.); }

    static scalar Trace(const Matrix2x2& m) {
        return m[0][0] + m[1][1];
    }

    static scalar Trace(const Matrix3x3& m) {
        return m[0][0] + m[1][1] + m[2][2];
    }

    static scalar Trace(const Matrix4x4& m) {
        return m[0][0] + m[1][1] + m[2][2] + m[3][3];
    }

    // tr(m²)
    static scalar SquaredTrace(const Matrix2x2& m) {
        return m[0][0] * m[0][0] + m[1][0] * m[0][1] * scalar(2.0) + m[1][1] * m[1][1];
    }

    static scalar SquaredTrace(const Matrix3x3& m) {
        return (m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2]) +
               (m[0][1] * m[1][0] + m[1][2] * m[2][1] + m[2][0] * m[0][2]) * scalar(2.);
    }

    static scalar SquaredTrace(const Matrix4x4& m) {
        return (m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2] + m[3][3] * m[3][3]) +
               (m[0][1] * m[1][0] + m[0][2] * m[2][0] + m[0][3] * m[3][0] + m[1][2] * m[2][1] + m[1][3] * m[3][1] + m[2][3] * m[3][2]) * scalar(2.);
    }

    // tr(m^T m)
    static scalar SquaredNorm(const Matrix2x2& m) {
        scalar st(0.); 
        for (unsigned int i = 0; i < 2; ++i)
            for (unsigned int j = 0; j < 2; ++j)
                st += m[i][j] * m[i][j];
        return st;
    }

    static scalar SquaredNorm(const Matrix3x3& m) {
        scalar st(0.);
        for (unsigned int i = 0; i < 3; ++i)
            for (unsigned int j = 0; j < 3; ++j)
                st += m[i][j] * m[i][j];
        return st;
    }

    static scalar SquaredNorm(const Matrix4x4& m) {
        scalar st(0.);
        for (unsigned int i = 0; i < 4; ++i)
            for (unsigned int j = 0; j < 4; ++j)
                st += m[i][j] * m[i][j];
        return st;
    }

};

struct Unit3D
{
    static Vector3 Zero() { return Vector3(0., 0, 0.); }
    static Vector3 right() { return Vector3(1., 0, 0.);}
    static Vector3 up() { return Vector3(0., 1, 0.); }
    static Vector3 forward() { return Vector3(0., 0, 1.); }
};
