#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Rendering/Camera.h"
#include "Manager/Input.h"
class CameraManager : public Behaviour
{
public:
    CameraManager(const Vector3 target = Unit3D::Zero(), scalar distance = 0.0)
        : _speed(1.0), _mouse_sensitivity(0.01), _zoom(25.)
    {
        Camera::Instance();
    }

    virtual void init() override {
        Camera* camera   = Camera::Instance_ptr();
        _init_camera_pos = -Unit3D::forward() * scalar(8.);
        camera->position() = _init_camera_pos;
        camera->set_fov(_zoom);
        camera->update_vp();
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
        //Vector3 direction = Unit3D::Zero();
        //if (Input::Loop(Key::A)) direction.x = -1;
        //if (Input::Loop(Key::D)) direction.x = 1;
        //if (Input::Loop(Key::S)) direction.z = -1;
        //if (Input::Loop(Key::Z)) direction.z = 1;
        //if (Input::Loop(Key::Q)) direction.y = -1;
        //if (Input::Loop(Key::E)) direction.y = 1;

        //camera->target() += direction * _speed * Time::DeltaTime();
        //camera->position() += direction * _speed * Time::DeltaTime();
        if (Input::Down(Key::R))
        {
            camera->position() = _init_camera_pos;
            camera->target()   = Unit3D::Zero();
        }
    }

    virtual void rotate_camera(Camera* camera)
    { 
        if (!Input::Loop(M_LEFT)) return;

        Vector2 offset = Input::MouseOffset() * _mouse_sensitivity;
        if (offset.x == 0 && offset.y == 0) return;
        
        Vector3 lookDir      = camera->position() - camera->target();
        Matrix4x4 x_rotate = glm::rotate(
            Matrix::Identity4x4(), offset.y,
            glm::normalize(glm::cross(lookDir, Unit3D::up())));
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
            = Vector3(y_rotate * Vector4(camera->position(), 0.f));

    }

    void zoom_camera(Camera* camera)
    {
        scalar scroll = Input::MouseScroll();

        if (scroll != 0)
        {
            _zoom -= (scalar)scroll;
            _zoom = std::max(_zoom, scalar(1.));
            _zoom = std::min(_zoom, scalar(45.));
            camera->set_fov(_zoom);
        }
        
    }


    virtual ~CameraManager() { Camera::Delete(); }

protected:
    Vector3 _init_camera_pos;
    scalar _speed;
    scalar _mouse_sensitivity;
    scalar _zoom;
};