#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Tools/Axis.h"
enum ProjectionType{ Persective, Orthographic };

class Camera : public Singleton<Camera>
{
protected:
    friend Singleton;
    Camera()
        : _far(1000.f)
        , _near(0.01f)
        , _fov(60.f)
        , _ratio(0.5f)
        , _type(Persective)
        , _target(Unit3D::Zero())
        , _up(Unit3D::up())
        , _position(Unit3D::Zero())
        , _projection(Matrix::Identity4x4())
    { }

public:
    void build(scalar near = 0.01f, scalar far = 1000.f, scalar fov = 60.f, ProjectionType type = Persective);

    void update_vp();

    void set_type(ProjectionType type) { _type = type;}
    void set_near(scalar near) { _near = near;}
    void set_far(scalar far) { _far = far;}
    void set_fov(scalar fov) { _fov = fov;}
    
    Vector3& position() { return _position; }
    Vector3& target() { return _target; }
    [[nodiscard]] scalar near() const { return _near; }
    [[nodiscard]] scalar far() const { return _far; }
    [[nodiscard]] scalar fov() const { return _fov; }

    [[nodiscard]] const Matrix4x4& view() const { return _view; }
    [[nodiscard]] const Matrix4x4& projection() const { return _projection; }

protected:
    scalar _near, _far, _fov, _ratio;
    Vector3 _target, _position, _up;
    Matrix4x4 _projection, _view{};
    ProjectionType _type;
};