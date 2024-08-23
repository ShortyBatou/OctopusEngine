#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Rendering/Camera.h"

class CameraManager final : public Behaviour {
public:
    explicit CameraManager(const Vector3 &init_pos = -Unit3D::forward() * 8.f, const Vector3 &target = Unit3D::Zero());

    void update() override;

    void move_camera(Camera *camera) const;

    void rotate_camera(Camera *camera) const;

    void zoom_camera(Camera *camera);

    scalar &speed() { return _speed; }
    scalar &zoom() { return _zoom; }
    Vector2 &zoom_range() { return _zoom_range; }

    ~CameraManager() override { Camera::Delete(); }

protected:
    Vector2 _zoom_range;
    Vector3 _init_camera_pos;
    scalar _speed;
    scalar _zoom;
};
