#include "Script/Dynamic/Cuda_Constraint_Rigid_Controller.h"
#include <Core/Entity.h>
#include <Manager/Debug.h>
#include <Manager/TimeManager.h>
#include <Script/Dynamic/Cuda_Dynamic_Test.h>
#include <Manager/Input.h>

void Cuda_Constraint_Rigid_Controller::late_init() {
    Cuda_Dynamic *cuda_dynamic = _entity->get_component<Cuda_Dynamic>();
    Mesh* mesh = _entity->get_component<Mesh>();
    _fixation = new GPU_Plane_Fix(mesh->geometry(), _plane_pos, _plane_normal);
    cuda_dynamic->add_dynamic(_fixation);
    _timer = 0;
    _smooth_step = _smooth_iterations;
}

void Cuda_Constraint_Rigid_Controller::update() {
    _fixation->active = _mode != -1;
    if(!_fixation->active) return;

    if (Input::Down(Key::NUM_0)) _mode = 0;
    if (Input::Down(Key::NUM_1)) _mode = 1;
    if (Input::Down(Key::NUM_2)) _mode = 2;
    if (Input::Down(Key::NUM_3)) _mode = 3;
    if (Input::Down(Key::NUM_4)) _mode = 4;
    if (Input::Down(Key::NUM_5)) {
        _mode = 5;
        _timer = 0;
    }
    if (Input::Down(Key::NUM_6)) {
        _mode = 6;
        _timer = 0;
    }
    if (Input::Down(Key::NUM_7)) {
        _mode = 7;
        _timer = 0;
    }
    if (Input::Down(Key::NUM_8)) {
        _mode = 8;
        _timer = 0;
    }

    if (Input::Down(Key::NUM_9) || _mode == 9) {
        _mode = 0;
        //rgn_crush();
    }


    if (_mode == 1) {
        Matrix4x4 rot = _fixation->rot;
        _fixation->rot = glm::rotate(rot, glm::radians(_rot_speed) * Time::Fixed_DeltaTime(), _plane_normal);
    }
    if (_mode == 2) {
        Matrix4x4 rot = _fixation->rot;
        _fixation->rot = glm::rotate(rot, glm::radians(-_rot_speed) * Time::Fixed_DeltaTime(), _plane_normal);
    }
    if (_mode == 3) {
        _fixation->offset += _plane_normal * Time::Fixed_DeltaTime() * _move_speed;
    }
    if (_mode == 4) {
        _fixation->offset -= _plane_normal * Time::Fixed_DeltaTime() * _move_speed;
    }
    if (_mode == 5) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            Matrix4x4 rot = _fixation->rot;
            _fixation->rot = glm::rotate(rot, glm::radians(_rot_speed / scalar(_smooth_iterations)), _plane_normal);
            _smooth_step++;
            _timer = 0;
        }
    }
    if (_mode == 6) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            Matrix4x4 rot = _fixation->rot;
            _fixation->rot = glm::rotate(rot, glm::radians(-_rot_speed / scalar(_smooth_iterations)), _plane_normal);
            _smooth_step++;
            _timer = 0;
        }
    }

    if (_mode == 7) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            _fixation->offset += _plane_normal * _move_speed / static_cast<scalar>(_smooth_iterations);
            _smooth_step++;
            _timer = 0;
        }
    }
    if (_mode == 8) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            _fixation->offset -= _plane_normal * _move_speed / static_cast<scalar>(_smooth_iterations);
            _smooth_step++;
            _timer = 0;
        }
    }
    Debug::Line(_fixation->com, _fixation->com + _fixation->normal);
}