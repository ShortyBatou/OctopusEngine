#pragma once
#include <random>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/VBD/VBD_Object.h"
struct VBD_FEM final : VBD_Object{
    VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape *shape,
            FEM_ContinuousMaterial *material, scalar k_damp = 0.f);

    void build_neighboors(const Mesh::Topology &topology);

    void build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry);

    void solve(ParticleSystem *ps, scalar dt) override;

    void plot_residual(ParticleSystem *ps, scalar dt);

    scalar compute_energy(ParticleSystem *ps) const;

    std::vector<Vector3> compute_forces(ParticleSystem *ps, scalar dt) const;

    void solve_vertex(ParticleSystem *ps, scalar dt, int vid);

    void solve_element(ParticleSystem *ps, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i);

    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);

    void compute_inertia(ParticleSystem *ps, scalar dt) override;

protected:
    scalar _k_damp;
    std::vector<Vector3> _y; // inertia
    Mesh::Topology _topology;
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice

    FEM_ContinuousMaterial *_material;
    FEM_Shape *_shape;
    std::vector<std::vector<Matrix3x3> > JX_inv; // per element
    std::vector<std::vector<scalar> > V; // per element
};

struct VertexBlockDescent final : ParticleSystem {
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
