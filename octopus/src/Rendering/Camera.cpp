#include "Rendering/Camera.h"
#include <UI/AppInfo.h>

void Camera::build(scalar near, scalar far, scalar fov, ProjectionType type)
{
    _near = near;
    _far  = far;
    _fov  = fov;
    _type = type;
    update_vp();
}

void Camera::update_vp()
{
    int w, h;
    AppInfo::Window_sizes(w, h);
    _ratio = static_cast<float>(w) / static_cast<float>(h);
    if (_type == Persective)
    {
        _projection
            = glm::perspective(glm::radians(_fov), _ratio, _near, _far);
    }
    else
    {
        const float right = static_cast<float>(w) * 0.005f;
        const float top   = static_cast<float>(h) * 0.005f;
        _projection = glm::ortho(-right, right, -top, top, _near, _far);
    }

    Vector3 lookDir = _target - _position;
    _up = glm::cross(lookDir, glm::vec3(0.f, 1.f, 0.f));
    _up = glm::cross(_up, lookDir);
    _up = glm::normalize(_up);

    _view = glm::lookAt(_position, _target, _up);
}
