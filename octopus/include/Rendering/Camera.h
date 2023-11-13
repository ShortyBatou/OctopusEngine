#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Tools/Axis.h"
#include "HUD/AppInfo.h"
enum ProjectionType{ Persective, Orthographic };

class Camera : public Singleton<Camera>
{
protected:
    friend Singleton<Camera>;
    Camera()
        : _far(1000.)
        , _near(0.01)
        , _fov(60.)
        , _ratio(0.5)
        , _type(Persective)
        , _target(Unit3D::Zero())
        , _up(Unit3D::up())
        , _position(Unit3D::Zero())
        , _projection(Matrix::Identity4x4())
    { }

public:
    void build(scalar near = 0.01, scalar far = 1000., scalar fov = 60., ProjectionType type = Persective)
    { 
        _near = near;
        _far  = far;
        _fov  = fov;
        _type = type;
        update_vp();
    }

    void update_vp()
    {
        unsigned int w, h;
        AppInfo::Window_sizes(w, h);
        _ratio = float(w) / float(h);
        if (_type == Persective)
        {
            _projection
                = glm::perspective(glm::radians(_fov), _ratio, _near, _far);
        }
        else
        {
            float right = float(w) * 0.005;
            float top   = float(h) * 0.005; 
            _projection = glm::ortho(-right, right, -top, top, _near, _far);
        }

        Vector3 lookDir = _target - _position;
        _up = glm::cross(lookDir, glm::vec3(0.f, 1.f, 0.f));
        _up = glm::cross(_up, lookDir);
        _up = glm::normalize(_up);

        _view = glm::lookAt(_position, _target, _up);
    }

    void set_type(ProjectionType type) { _type = type;}
    void set_near(scalar near) { _near = near;}
    void set_far(scalar far) { _far = far;}
    void set_fov(scalar fov) { _fov = fov;}
    
    Vector3& position() { return _position; }
    Vector3& target() { return _target; }
    scalar near() const { return _near; }
    scalar far() const { return _far; }
    scalar fov() const { return _fov; }

    const Matrix4x4& view() { return _view; }
    const Matrix4x4& projection() { return _projection; }

protected:
    scalar _near, _far, _fov, _ratio;
    Vector3 _target, _position, _up;
    Matrix4x4 _projection, _view;
    ProjectionType _type;
};