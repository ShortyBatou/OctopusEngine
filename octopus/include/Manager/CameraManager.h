#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Rendering/Camera.h"
#include "Manager/Input.h"
class CameraManager : public Behaviour
{
public:
    CameraManager(const Vector3& init_pos = -Unit3D::forward() * scalar(8.), const Vector3& target = Unit3D::Zero(), scalar distance = 0.0)
        : _speed(0.5), _zoom(45.), _init_camera_pos(init_pos), _zoom_range(Vector2(1., 90.))
    {
        Camera* camera = Camera::Instance_ptr();
        camera->position() = _init_camera_pos;
        camera->target() = target;
        camera->set_fov(_zoom);
        camera->update_vp();
    }

    virtual void init() override {
        
    }

    virtual void update() override { 
        Camera* camera = Camera::Instance_ptr();
        move_camera(camera);
        rotate_camera(camera);
        zoom_camera(camera);
        camera->update_vp();
    }

    virtual void move_camera(Camera* camera)
    {
        if (!Input::Loop(M_RIGHT)) return;
        Vector2 offset = Input::MouseOffset() * _speed * scalar(0.01);
        if (offset.x == 0 && offset.y == 0) return;

        Vector3 direction = Unit3D::Zero();
        Vector3 front = glm::normalize(camera->position() - camera->target());
        Vector3 right = glm::cross(front, Unit3D::up());
        Vector3 up = glm::cross(front, right);
        direction = (-up * offset.y + right * offset.x);

        camera->target() += direction;
        camera->position() += direction;
        if (Input::Down(Key::R))
        {
            camera->position() = _init_camera_pos;
            camera->target()   = Unit3D::Zero();
        }
    }

    virtual void rotate_camera(Camera* camera)
    { 
        if (!Input::Loop(M_LEFT)) return;

        Vector2 offset = Input::MouseOffset() * _speed * scalar(0.01);
        if (offset.x == 0 && offset.y == 0) return;
        camera->position() = camera->position() - camera->target();
        Vector3 lookDir      = camera->position();
        Matrix4x4 x_rotate = glm::rotate( Matrix::Identity4x4(), offset.y, glm::normalize(glm::cross(lookDir, Unit3D::up())));
        Vector3 rotv = glm::normalize(glm::cross(lookDir, Unit3D::up()));
        
        Vector3 p_temp(x_rotate * Vector4(camera->position(), 0.f));

        scalar sign_x = glm::dot(p_temp, Unit3D::right());
        sign_x *= glm::dot(camera->position(), Unit3D::right());
        scalar sign_z = glm::dot(p_temp, Unit3D::forward());
        sign_z *= glm::dot(camera->position(), Unit3D::forward());

        if (sign_x > 0 && sign_z > 0) camera->position() = p_temp;

        Matrix4x4 y_rotate = glm::rotate(Matrix::Identity4x4(), -offset.x,
                                            Unit3D::up());
        camera->position()
            = camera->target() + Vector3(y_rotate * Vector4(camera->position(), 0.f));

    }

    void zoom_camera(Camera* camera)
    {
        scalar scroll = Input::MouseScroll();

        if (scroll != 0)
        {
            _zoom -= (scalar)scroll;
            _zoom = std::max(_zoom, scalar(1.));
            _zoom = std::min(_zoom, scalar(90.));
            
        }
        camera->set_fov(_zoom);
        
    }

    scalar& speed() { return _speed; }
    scalar& zoom() { return _zoom; }
    Vector2& zoom_range() { return _zoom_range; }

    virtual ~CameraManager() { Camera::Delete(); }

protected:
    Vector2 _zoom_range;
    Vector3 _init_camera_pos;
    scalar _speed;
    scalar _zoom;
};