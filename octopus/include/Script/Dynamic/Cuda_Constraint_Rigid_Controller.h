#pragma once
#include "Core/Base.h"
#include "Core/Component.h"
#include "Script/Dynamic/Cuda_Dynamic_Test.h"
#include "GPU/GPU_PBD.h"

class Cuda_Constraint_Rigid_Controller final : public Component {
public:
	Cuda_Constraint_Rigid_Controller(const Vector3& p, const Vector3& n, const int mode = 0)
	: _mode(mode), _event_rate(1), _move_speed(1.), _rot_speed(90), _smooth_iterations(1), _smooth_step(1), _timer(0.f),
	_plane_pos(p),_plane_normal(n),_fixation(nullptr) { }

	void late_init() override;

	void update() override;

	void rgn_crush();

	~Cuda_Constraint_Rigid_Controller() override = default;

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
	GPU_Plane_Fix* _fixation;
};