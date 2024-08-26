#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>

using scalar = float;
const scalar min_limit = std::numeric_limits<scalar>::min();
const scalar max_limit = std::numeric_limits<scalar>::max();
const scalar eps = min_limit * static_cast<scalar>(1e5);
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

};

struct Unit3D
{
    static Vector3 Zero() { return {0., 0, 0.}; }
    static Vector3 right() { return {1., 0, 0.};}
    static Vector3 up() { return {0., 1, 0.}; }
    static Vector3 forward() { return {0., 0, 1.}; }
};
