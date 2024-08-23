#include "Script/Dynamic/ConstantForce_Controller.h"
#include "Script/Dynamic/ParticleSystemDynamic.h"
#include "Manager/Debug.h"

void ConstantForce_Controller::late_init() {
	ParticleSystemDynamic* ps_dynamic = _entity->get_component<ParticleSystemDynamic>();
	ParticleSystem* ps = ps_dynamic->getParticleSystem();
	std::vector<int> ids;
	for (int i = 0; i < ps->particles().size(); ++i) {
		Particle* part = ps->particles()[i];
		if (check_in_box(part->position)) {
			ids.push_back(i);
		}
 	}
	_cf = new ConstantForce(ids, _force);
	ps->add_constraint(_cf);
}

bool ConstantForce_Controller::check_in_box(Vector3& p) const {
	return p.x > _pmin.x && p.y > _pmin.y && p.z > _pmin.z
		&& p.x <= _pmax.x && p.y <= _pmax.y && p.z <= _pmax.z;
}

void ConstantForce_Controller::update() {
	Debug::SetColor(ColorBase::Red());
	Debug::Cube(_pmin, _pmax);
}
