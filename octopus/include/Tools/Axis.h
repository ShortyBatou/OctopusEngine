#pragma once
#include "Core/Base.h"
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>
struct Axis
{
    Axis(const Vector3 pos = Vector3(0.), const Matrix3x3 rot = Matrix::Identity3x3())
    {  
        setTransform(pos, rot);
    }

    void move(const Vector3& m)
    {
        _pos += m;
        updateTransform();
    }

    void setTransform(const Vector3& pos, const Matrix3x3& rot) { 
        _rot = rot; _pos = pos;
        updateTransform();
    }

    void setTransform(const Matrix4x4& t) {
        glm::quat rotation;
        glm::vec3 scale, translation, skew;
        glm::vec4 perspective;
        glm::decompose(t, scale, rotation, translation, skew, perspective);
        _rot = glm::toMat3(rotation);
        _pos = translation;
        updateTransform();
    }

    Matrix4x4 t() const { return _t; }
    void setRotation(const Matrix3x3& rot) { setTransform(_pos, rot); }
    void setPosition(const Vector3& position) { setTransform(position, _rot); }
    const Matrix4x4& rotationMatrix() const { return _rot; }
    const Vector3& position() const { return _pos; }
    const Vector3& normal() const { return _forward; }
    const Vector3& right() const { return _right; }
    const Vector3& up() const { return _up; }
    const Vector3& forward() const { return _forward; }

protected:
    void updateTransform() { 
        _t = _rot;
        _t = glm::translate(_t, _pos);

        _up = _rot * Unit3D::up();
        _right = _rot * Unit3D::right();
        _forward = _rot * Unit3D::forward();
    }

    Matrix4x4 _t;
    Matrix3x3 _rot;
    Vector3 _pos, _up, _right, _forward;
};
