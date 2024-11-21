#pragma once
#include <random>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/VBD/VBD_Object.h"
#include "Dynamic/VBD/VertexBlockDescent.h"

struct VBD_FEM final : VBD_Object {
    VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape *shape,
            FEM_ContinuousMaterial *material, scalar k_damp = 0.f);

    void compute_inertia(VertexBlockDescent* vbd, scalar dt) override;

    void build_neighboors(const Mesh::Topology &topology);

    void build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry);

    void solve(VertexBlockDescent *vbd, scalar dt) override;

    void plot_residual(VertexBlockDescent *vbd, scalar dt);

    scalar compute_energy(VertexBlockDescent *vbd) const;

    std::vector<Vector3> compute_forces(VertexBlockDescent *vbd, scalar dt) const;

    void solve_vertex(VertexBlockDescent *vbd, scalar dt, int vid);

    void solve_element(VertexBlockDescent *vbd, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i);

    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);


protected:
    scalar _k_damp;
    Mesh::Topology _topology;
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice
    std::vector<Vector3> _y; // inertia
    FEM_ContinuousMaterial *_material;
    FEM_Shape *_shape;
    std::vector<std::vector<Matrix3x3> > JX_inv; // per element
    std::vector<std::vector<scalar> > V; // per element
};
