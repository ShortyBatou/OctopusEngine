#pragma once
#include <random>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/VBD/VBD_Object.h"

struct VertexBlockDescent : ParticleSystem {
    explicit VertexBlockDescent(Solver *solver, const int iteration, const int sub_iteration, const scalar rho)
        : ParticleSystem(solver), _iteration(iteration), _sub_iteration(sub_iteration), _rho(rho) {
    }

    [[nodiscard]] scalar compute_omega(const scalar omega, const int it) const {
        if (it == 0) return 1.f;
        if (it == 1) return 2.f / (2.f - _rho * _rho);
        return 4.f / (4.f - _rho * _rho * omega);
    }

    void chebyshev_acceleration(int it, scalar &omega);

    void step(scalar dt) override;

    void update_velocity(scalar dt) const;

    void add(VBD_Object *obj) { _objs.push_back(obj); }

    ~VertexBlockDescent() override {
        for(const VBD_Object* obj : _objs) delete obj;
    }
protected:
    std::vector<VBD_Object *> _objs;
    std::vector<Vector3> prev_x;
    std::vector<Vector3> prev_prev_x;

    int _iteration;
    int _sub_iteration;
    scalar _rho;
};
