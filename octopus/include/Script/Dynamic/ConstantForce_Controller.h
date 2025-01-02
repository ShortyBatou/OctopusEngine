#pragma once
#include "Core/Base.h"
#include "Core/Entity.h"
#include "Dynamic/Base/Constraint.h"

class ConstantForce_Controller : public Component {
public:
	ConstantForce_Controller(const Vector3& pmin, const Vector3& pmax, const Vector3& force)
	: _force(force), _multiplier(0.f), _pmin(pmin), _pmax(pmax),_fixation(nullptr), _cf(nullptr)  { }

	void late_init() override;

	bool check_in_box(Vector3& p) const;

	void update() override;

	Vector3 _force;
	scalar _multiplier;
protected:
	Vector3 _pmin, _pmax;
	RB_Fixation* _fixation;
	ConstantForce* _cf;
};