#pragma once

#include <ostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

using scalar = float;
constexpr scalar min_limit = std::numeric_limits<scalar>::min();
constexpr scalar max_limit = std::numeric_limits<scalar>::max();
constexpr scalar eps = min_limit * static_cast<scalar>(1e6);
constexpr scalar small_eps = 1e-12;
constexpr scalar large_eps = 1e-6;
constexpr glm::precision precision = glm::precision::defaultp;
const scalar PI = glm::pi<scalar>();

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

namespace Matrix
{
    inline Matrix2x2 Zero2x2() { return Matrix2x2(0.); }
    inline Matrix3x3 Zero3x3() { return Matrix3x3(0.); }
    inline Matrix4x4 Zero4x4() { return Matrix4x4(0.); }

    inline Matrix2x2 Identity2x2() { return Matrix2x2(1.); }
    inline Matrix3x3 Identity3x3() { return Matrix3x3(1.); }
    inline Matrix4x4 Identity4x4() { return Matrix4x4(1.); }

    scalar Trace(const Matrix2x2& m);
    scalar Trace(const Matrix3x3& m);
    scalar Trace(const Matrix4x4& m);

    scalar SquaredTrace(const Matrix2x2& m);
    scalar SquaredTrace(const Matrix3x3& m);
    scalar SquaredTrace(const Matrix4x4& m);
    scalar SquaredNorm(const Matrix2x2& m);
    scalar SquaredNorm(const Matrix3x3& m);
    scalar SquaredNorm(const Matrix4x4& m);

    Matrix3x3 Hat(const Vector3& v);
    Matrix3x3 Com(const Matrix3x3& m);
    void PolarDecomposition(const Matrix3x3& m, Matrix3x3& R, Matrix3x3& S);
    void PolarDecompositionOpti(const glm::mat3& F, glm::mat3& R, glm::mat3& S, int maxIter = 10, float tol = 1e-6);
    Vector3 SingularValues(const Matrix3x3& m);
};

inline std::ostream& operator<<(std::ostream& os, const Vector2& v) {
    os << "(x:" << v.x << " y:" << v.y << ')';
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Vector3& v) {
    os << "(x:" << v.x << " y:" << v.y << " z:" << v.z <<  ')';
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Matrix2x2& m) {
    os << '|' << m[0][0] << "\t" << m[0][1] << "|\n|"
              << m[1][0] << "\t" << m[1][1] << '|';
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Matrix3x3& m) {
    os << '|' << m[0][0] << "\t" << m[0][1] << "\t" << m[0][2] << "|\n|"
              << m[1][0] << "\t" << m[1][1] << "\t" << m[1][2] << "|\n|"
              << m[2][0] << "\t" << m[2][1] << "\t" << m[2][2] << '|';
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Matrix4x4& m) {
    os << '|' << m[0][0] << "\t" << m[0][1] << "\t" << m[0][2] << "\t" << m[0][3] << "|\n|"
              << m[1][0] << "\t" << m[1][1] << "\t" << m[1][2] << "\t" << m[1][3] << "|\n|"
              << m[2][0] << "\t" << m[2][1] << "\t" << m[2][2] << "\t" << m[2][3] << "|\n|"
              << m[3][0] << "\t" << m[3][1] << "\t" << m[3][2] << "\t" << m[3][3] << '|';
    return os;
}


struct Unit3D
{
    static Vector3 Zero() { return {0., 0, 0.}; }
    static Vector3 right() { return {1., 0, 0.};}
    static Vector3 up() { return {0., 1, 0.}; }
    static Vector3 forward() { return {0., 0, 1.}; }
};
