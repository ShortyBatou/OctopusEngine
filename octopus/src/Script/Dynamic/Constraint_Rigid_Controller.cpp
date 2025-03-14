#include "Script/Dynamic/Constraint_Rigid_Controller.h"
#include <Manager/TimeManager.h>

void Constraint_Rigid_Controller::late_init() {
    ParticleSystemDynamic *ps_dynamic = this->_entity->get_component<ParticleSystemDynamic>();
    ParticleSystem *ps = ps_dynamic->getParticleSystem();
    std::vector<int> ids;
    for (int i = 0; i < ps->particles().size(); ++i) {
        const Particle *part = ps->particles()[i];
        const Vector3 dir = part->init_position - _plane_pos + _plane_normal * 0.001f;
        if (glm::dot(dir, _plane_normal) >= 0) {
            ids.push_back(i);
        }
    }
    _fixation = new RB_Fixation(ids);
    _plane = new PlaneConstraint(_plane_pos, -_plane_normal);
    ps->add_constraint(_fixation);
    //ps->add_effect(_plane);
    _plane->set_active(false);
    _timer = 0;
    _smooth_step = _smooth_iterations;
}

void Constraint_Rigid_Controller::update() {
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
        rgn_crush();
    }

    _fixation->set_active(_mode != -1);
    if (_mode == 1) {
        const Matrix4x4 rot = _fixation->rot;
        _fixation->rot = glm::rotate(rot, glm::radians(_rot_speed) * Time::Fixed_DeltaTime(), _plane_normal);
    }
    if (_mode == 2) {
        const Matrix4x4 rot = _fixation->rot;
        _fixation->rot = glm::rotate(rot, glm::radians(-_rot_speed) * Time::Fixed_DeltaTime(), _plane_normal);
    }
    if (_mode == 3) {
        _fixation->offset += _plane_normal * Time::Fixed_DeltaTime() * _move_speed;
        _plane->_o = _fixation->com + _fixation->offset;
    }
    if (_mode == 4) {
        _fixation->offset -= _plane_normal * Time::Fixed_DeltaTime() * _move_speed;
        _plane->_o = _fixation->com + _fixation->offset;
    }
    if (_mode == 5) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            const Matrix4x4 rot = _fixation->rot;
            _fixation->rot = glm::rotate(rot, glm::radians(_rot_speed / static_cast<scalar>(_smooth_iterations)), _plane_normal);
            _smooth_step++;
            _timer = 0;
        }
    }
    if (_mode == 6) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            const Matrix4x4 rot = _fixation->rot;
            _fixation->rot = glm::rotate(rot, glm::radians(-_rot_speed / static_cast<scalar>(_smooth_iterations)), _plane_normal);
            _smooth_step++;
            _timer = 0;
        }
    }

    if (_mode == 7) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            _fixation->offset += _plane_normal * _move_speed / static_cast<scalar>(_smooth_iterations);
            _plane->_o = _fixation->com + _fixation->offset / static_cast<scalar>(_smooth_iterations);
            _smooth_step++;
            _timer = 0;
        }
    }
    if (_mode == 8) {
        _timer += Time::Fixed_DeltaTime();
        if (_timer >= _event_rate) _smooth_step = 0;
        if (_smooth_step < _smooth_iterations) {
            _fixation->offset -= _plane_normal * _move_speed / static_cast<scalar>(_smooth_iterations);
            _plane->_o = _fixation->com + _fixation->offset / static_cast<scalar>(_smooth_iterations);
            _smooth_step++;
            _timer = 0;
        }
    }
}

void Constraint_Rigid_Controller::rgn_crush() const {
    ParticleSystemDynamic *ps_dynamic = this->_entity->get_component<ParticleSystemDynamic>();
    ParticleSystem *ps = ps_dynamic->getParticleSystem();
    for (const auto part: ps->particles()) {
        part->position.y = _plane_pos.y;
        /*part->position.x = _plane_pos.x + (scalar(rand()) / scalar(RAND_MAX)) * 2.f - 1.f;
        part->position.z = _plane_pos.z + (scalar(rand()) / scalar(RAND_MAX)) * 2.f - 1.f;*/
        part->last_position = part->position;
        part->velocity = Unit3D::Zero();
    }
    ps_dynamic->update_mesh();
}
