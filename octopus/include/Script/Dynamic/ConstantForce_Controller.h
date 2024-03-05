#pragma once
#include "Core/Base.h"
#include "ParticleSystemDynamic.h"
#include "Core/Entity.h"
#include "Dynamic/Base/Constraint.h"
#include <Tools/Color.h>

class ConstantForce_Controller : public Component {
public:
	ConstantForce_Controller(const Vector3& pmin, const Vector3& pmax, const Vector3& force) : _force(force), _pmin(pmin), _pmax(pmax) { }

	virtual void late_init() override {
		ParticleSystemDynamic* ps_dynamic = this->_entity->getComponent<ParticleSystemDynamic>();
		ParticleSystem* ps = ps_dynamic->getParticleSystem();
		std::vector<unsigned int> ids;
		for (unsigned int i = 0; i < ps->particles().size(); ++i) {
			Particle* part = ps->particles()[i];
			if (check_in_box(part->position)) {
				ids.push_back(i);
			}
 		}
		_cf = new ConstantForce(ids, _force);
		ps->add_constraint(_cf);
	}

	bool check_in_box(Vector3& p) {
		return p.x > _pmin.x && p.y > _pmin.y && p.z > _pmin.z
			&& p.x <= _pmax.x && p.y <= _pmax.y && p.z <= _pmax.z;
	}

	virtual void update() override {
		Debug::SetColor(ColorBase::Red());
		Debug::Cube(_pmin, _pmax);
	}

public:
	Vector3 _force;
	scalar _multiplier;

protected:
	Vector3 _pmin, _pmax;
	RB_Fixation* _fixation;
	ConstantForce* _cf;
};