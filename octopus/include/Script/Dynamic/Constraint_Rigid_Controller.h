#pragma once
#include "Core/Base.h"
#include "Core/Component.h"
#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Dynamic/Base/Constraint.h"
class Constraint_Rigid_Controller : public Component {
public:
	Constraint_Rigid_Controller(const Vector3& p, const Vector3& n, const int mode = 0)
	: _mode(mode), _event_rate(1), _move_speed(0.5), _rot_speed(45), _smooth_iterations(1), _smooth_step(1), _timer(0.f),
	_plane_pos(p),_plane_normal(n),_fixation(nullptr), _plane(nullptr) { }

	void late_init() override;

	void update() override;

	void rgn_crush();

	int _mode;
	scalar _event_rate;
	scalar _move_speed;
	scalar _rot_speed;
	int _smooth_iterations;
	
protected:
	int _smooth_step;
	scalar _timer;
	Vector3 _plane_pos;
	Vector3 _plane_normal;
	RB_Fixation* _fixation;
	PlaneConstraint* _plane;
};