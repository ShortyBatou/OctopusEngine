#pragma once
#include "Core/Base.h"
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
struct Axis
{
    explicit Axis(const Vector3& pos = Vector3(0.), const Matrix3x3 &rot = Matrix::Identity3x3()) :
     _t({}), _rot({}), _pos({}), _up({}), _right({}), _forward({})
    {  
        setTransform(pos, rot);
    }

    void move(const Vector3& m);

    void setTransform(const Vector3& pos, const Matrix3x3& rot);

    void setTransform(const Matrix4x4& t);

    [[nodiscard]] Matrix4x4 t() const { return _t; }
    void setRotation(const Matrix3x3& rot) { setTransform(_pos, rot); }
    void setPosition(const Vector3& position) { setTransform(position, _rot); }
    [[nodiscard]] Matrix3x3 rotation() const { return _rot; }
    [[nodiscard]] const Matrix4x4& rotation4x4() const { return _rot; }
    [[nodiscard]] const Vector3& position() const { return _pos; }
    [[nodiscard]] const Vector3& normal() const { return _forward; }
    [[nodiscard]] const Vector3& right() const { return _right; }
    [[nodiscard]] const Vector3& up() const { return _up; }
    [[nodiscard]] const Vector3& forward() const { return _forward; }

protected:
    void updateTransform();

    Matrix4x4 _t;
    Matrix3x3 _rot;
    Vector3 _pos, _up, _right, _forward;
};
