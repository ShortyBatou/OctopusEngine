#pragma once
#include <GPU/GPU_Constraint.h>
#include <GPU/GPU_FEM.h>
#include "Core/Base.h"
#include "Core/Component.h"
#include "Tools/Area.h"
class Cuda_Constraint_Rigid_Controller final : public Component {
public:
	Cuda_Constraint_Rigid_Controller(Area* area, const Vector3& n, const int mode = 0)
		: _mode(mode), _event_rate(1), _move_speed(1.), _rot_speed(90), _smooth_iterations(1), _area(area),
		  _smooth_step(1),
		  _timer(0.f), _rot_normal(n), _fixation(nullptr), _crush(nullptr), _random_sphere(nullptr) {
	}

	void late_init() override;

	void update() override;

	void rgn_crush();

	~Cuda_Constraint_Rigid_Controller() override {
		delete _area;
	}

	int _mode;
	scalar _event_rate;
	scalar _move_speed;
	scalar _rot_speed;
	int _smooth_iterations;

protected:
	Area* _area;
	int _smooth_step;
	scalar _timer;
	Vector3 _rot_normal;
	GPU_Fix_Constraint* _fixation;
	GPU_Crush* _crush;
	GPU_RandomSphere* _random_sphere;
};