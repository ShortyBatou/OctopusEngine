#include "Tools/Axis.h"

void Axis::move(const Vector3& m)
{
    _pos += m;
    updateTransform();
}

void Axis::setTransform(const Vector3& pos, const Matrix3x3& rot) {
    _rot = rot; _pos = pos;
    updateTransform();
}

void Axis::setTransform(const Matrix4x4& t) {
    glm::quat rotation;
    glm::vec3 scale, translation, skew;
    glm::vec4 perspective;
    glm::decompose(t, scale, rotation, translation, skew, perspective);
    _rot = glm::toMat3(rotation);
    _pos = translation;
    updateTransform();
}


void Axis::updateTransform() {
    _t = _rot;
    _t = glm::translate(_t, _pos);

    _up = _rot * Unit3D::up();
    _right = _rot * Unit3D::right();
    _forward = _rot * Unit3D::forward();
}

