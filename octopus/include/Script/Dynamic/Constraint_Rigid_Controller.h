#pragma once
#include "Core/Base.h"
#include "Core/Component.h"
#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Dynamic/Base/Constraint.h"
class Constraint_Rigid_Controller : public Component {
public:
	Constraint_Rigid_Controller(const Vector3& p, const Vector3& n, unsigned int mode = 0) : _plane_pos(p), _plane_normal(n), _mode(mode), _rot_speed(45), _move_speed(0.5) { }

	virtual void late_init() override {
		ParticleSystemDynamic* ps_dynamic = this->_entity->getComponent<ParticleSystemDynamic>();
		ParticleSystem* ps = ps_dynamic->getParticleSystem();
		std::vector<unsigned int> ids;
		for (unsigned int i = 0; i < ps->particles().size(); ++i) {
			Particle* part = ps->particles()[i];
			Vector3 dir = part->init_position - _plane_pos;
			if (glm::dot(dir, _plane_normal) >= 0) 
			{
				ids.push_back(i);
			}
		}
		_fixation = new RB_Fixation(ids);
		ps->add_constraint(_fixation);
		_timer = 0;
	}

	virtual void update() override {
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
			ParticleSystemDynamic* ps_dynamic = this->_entity->getComponent<ParticleSystemDynamic>();
			ParticleSystem* ps = ps_dynamic->getParticleSystem();
			for (unsigned int i = 0; i < ps->particles().size(); ++i) {
				Particle* part = ps->particles()[i];
				part->position.y = _plane_pos.y;
				part->position.x = _plane_pos.x + (scalar(rand()) / scalar(RAND_MAX)) * 2.f - 1.f;
				part->position.z = _plane_pos.z + (scalar(rand()) / scalar(RAND_MAX)) * 2.f - 1.f;
				part->last_position = part->position;
				part->velocity = Unit3D::Zero();
			}
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
			if (_timer >= 1.) {
				Matrix4x4 rot = _fixation->rot;
				_fixation->rot = glm::rotate(rot, glm::radians(_rot_speed), _plane_normal);
				_timer = 0;
			}
		}
		if (_mode == 6) {
			_timer += Time::Fixed_DeltaTime();
			if (_timer >= 1.) {
				Matrix4x4 rot = _fixation->rot;
				_fixation->rot = glm::rotate(rot, glm::radians(-_rot_speed), _plane_normal);
				_timer = 0;
			}
		}

		if (_mode == 7) {
			_timer += Time::Fixed_DeltaTime();
			if (_timer >= 1.) {
				_fixation->offset += _plane_normal * _move_speed;
				_timer = 0;
			}
		}
		if (_mode == 8) {
			_timer += Time::Fixed_DeltaTime();
			if (_timer >= 1.) {
				_fixation->offset -= _plane_normal * _move_speed;
				_timer = 0;
			}
		}
	}

protected:
	scalar _timer;
	scalar _move_speed;
	scalar _rot_speed;
	unsigned int _mode;
	Vector3 _plane_pos;
	Vector3 _plane_normal;
	RB_Fixation* _fixation;
};