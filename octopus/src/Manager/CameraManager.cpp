#pragma once
#include "Manager/CameraManager.h"
#include "Manager/Input.h"

CameraManager::CameraManager(const Vector3 &init_pos, const Vector3 &target): _zoom_range(Vector2(1., 90.)),
                                                                              _init_camera_pos(init_pos), _speed(0.5),
                                                                              _zoom(45.) {
    Camera *camera = Camera::Instance_ptr();
    camera->position() = _init_camera_pos;
    camera->target() = target;
    camera->set_fov(_zoom);
    camera->update_vp();
}


void CameraManager::update() {
    Camera *camera = Camera::Instance_ptr();
    move_camera(camera);
    rotate_camera(camera);
    zoom_camera(camera);
    camera->update_vp();
}

void CameraManager::move_camera(Camera *camera) const {
    if (!Input::Loop(M_RIGHT)) return;
    const Vector2 offset = Input::MouseOffset() * _speed * scalar(0.01);
    if (offset.x == 0 && offset.y == 0) return;

    const Vector3 front = glm::normalize(camera->position() - camera->target());
    const Vector3 right = glm::cross(front, Unit3D::up());
    const Vector3 up = glm::cross(front, right);
    const Vector3 direction = (-up * offset.y + right * offset.x);

    camera->target() += direction;
    camera->position() += direction;
    if (Input::Down(Key::R)) {
        camera->position() = _init_camera_pos;
        camera->target() = Unit3D::Zero();
    }
}

void CameraManager::rotate_camera(Camera *camera) const {
    if (!Input::Loop(M_LEFT)) return;

    const Vector2 offset = Input::MouseOffset() * _speed * 0.01f;
    if (offset.x == 0 && offset.y == 0) return;
    camera->position() = camera->position() - camera->target();
    const Vector3 lookDir = camera->position();
    const Matrix4x4 x_rotate = glm::rotate(Matrix::Identity4x4(), offset.y,
                                           glm::normalize(glm::cross(lookDir, Unit3D::up())));

    const Vector3 p_temp(x_rotate * Vector4(camera->position(), 0.f));

    scalar sign_x = glm::dot(p_temp, Unit3D::right());
    sign_x *= glm::dot(camera->position(), Unit3D::right());
    scalar sign_z = glm::dot(p_temp, Unit3D::forward());
    sign_z *= glm::dot(camera->position(), Unit3D::forward());

    if (sign_x > 0 && sign_z > 0) camera->position() = p_temp;

    const Matrix4x4 y_rotate = glm::rotate(Matrix::Identity4x4(), -offset.x,
                                           Unit3D::up());
    camera->position()
            = camera->target() + Vector3(y_rotate * Vector4(camera->position(), 0.f));
}

void CameraManager::zoom_camera(Camera *camera) {
    if (const scalar scroll = Input::MouseScroll(); scroll != 0) {
        _zoom -= scroll;
        _zoom = std::max(_zoom, 1.f);
        _zoom = std::min(_zoom, 90.f);
    }
    camera->set_fov(_zoom);
}
