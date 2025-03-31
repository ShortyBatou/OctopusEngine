#include "Script/Dynamic/Cuda_Constraint_Rigid_Controller.h"
#include <Core/Entity.h>
#include <Manager/Debug.h>
#include <Manager/TimeManager.h>
#include <Script/Dynamic/Cuda_XPBD_FEM_Dynamic.h>
#include <GPU/GPU_Constraint.h>
#include <Manager/Input.h>

void Cuda_Constraint_Rigid_Controller::late_init() {
    const Cuda_ParticleSystem_Dynamics *cuda_dynamic = _entity->get_component<Cuda_ParticleSystem_Dynamics>();
    Mesh* mesh = _entity->get_component<Mesh>();
    _fixation = new GPU_Fix_Constraint(mesh->geometry(), _area);
    _crush = new GPU_Crush(); _crush->active = false;
    _random_sphere = new GPU_RandomSphere(Vector3(0.,1.,0.), 1.); _random_sphere->active = false;
    cuda_dynamic->get_particle_system()->add_constraint(_fixation);
    cuda_dynamic->get_particle_system()->add_constraint(_crush);
    cuda_dynamic->get_particle_system()->add_constraint(_random_sphere);
    cuda_dynamic->get_particle_system()->add_constraint(new GPU_Box_Limit(Vector3(-10,-4,-10), Vector3(10,4,10)));
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

    if(Input::Up(Key::NUM_9))
    {
        _mode = 0;
        _crush->active = false;
    }

    if (Input::Down(Key::NUM_9)) {
        _mode = 9;
        _crush->active = true;
    }

    if(Input::Up(Key::P))
    {
        _mode = 0;
        _random_sphere->active = false;
    }

    if (Input::Down(Key::P)) {
        _mode = 10;
        _random_sphere->active = true;
    }


    if (_mode == 1) {
        const Matrix4x4 rot = _fixation->axis.rotation4x4();
        _fixation->axis.setRotation(glm::rotate(rot, glm::radians(_rot_speed) * Time::Fixed_DeltaTime(), _rot_normal));
    }
    if (_mode == 2) {
        const Matrix4x4 rot = _fixation->axis.rotation4x4();
        _fixation->axis.setRotation(glm::rotate(rot, glm::radians(-_rot_speed) * Time::Fixed_DeltaTime(), _rot_normal));
    }
    if (_mode == 3) {
        _fixation->axis.move(_rot_normal * Time::Fixed_DeltaTime() * _move_speed);
    }
    if (_mode == 4) {
        _fixation->axis.move(-_rot_normal * Time::Fixed_DeltaTime() * _move_speed);
    }
    if (_mode == 5) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            const Matrix4x4 rot = _fixation->axis.rotation4x4();
            _fixation->axis.setRotation(glm::rotate(rot, glm::radians(_rot_speed / static_cast<scalar>(_smooth_iterations)), _rot_normal));
            _smooth_step++;
            _timer = 0;
        }
    }
    if (_mode == 6) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            const Matrix4x4 rot = _fixation->axis.rotation4x4();
            _fixation->axis.setRotation(glm::rotate(rot, glm::radians(-_rot_speed / static_cast<scalar>(_smooth_iterations)), _rot_normal));
            _smooth_step++;
            _timer = 0;
        }
    }

    if (_mode == 7) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            _fixation->axis.move(_rot_normal * _move_speed / static_cast<scalar>(_smooth_iterations));
            _smooth_step++;
            _timer = 0;
        }
    }
    if (_mode == 8) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            _fixation->axis.move(-_rot_normal * _move_speed / static_cast<scalar>(_smooth_iterations));
            _smooth_step++;
            _timer = 0;
        }
    }
    if(_mode == 9) {
        _crush->active = true;
    }
    else {
        _crush->active = false;
    }
    if(_mode == 10) {
        _random_sphere->active = true;
    }
    else {
        _random_sphere->active = false;
    }
    Debug::SetColor(ColorBase::Red());
    _area->draw();
}