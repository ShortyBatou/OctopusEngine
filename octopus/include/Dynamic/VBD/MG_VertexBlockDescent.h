#pragma once
#include <random>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"

struct Grid_Level {
    Mesh::Geometry positions;
    std::vector<scalar> masses;
    std::vector<std::vector<Matrix3x3> > JX_inv;
    std::vector<std::vector<scalar> > V;
    FEM_Shape *_shape;
};

struct MG_VBD_FEM {
    MG_VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape *shape, FEM_ContinuousMaterial *material);

    void build_neighboors(const Mesh::Topology &topology);

    void build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry);

    void solve(ParticleSystem *ps, scalar dt);

    void plot_residual(ParticleSystem *ps, scalar dt);

    scalar compute_energy(ParticleSystem *ps) const;

    std::vector<Vector3> compute_forces(ParticleSystem *ps, scalar dt) const;

    void solve_vertex(ParticleSystem *ps, scalar dt, int vid);

    void solve_element(ParticleSystem *ps, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i);

    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);

    void compute_inertia(ParticleSystem *ps, scalar dt);

protected:
    std::vector<Vector3> _y; // inertia
    Mesh::Topology _topology;
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice

    FEM_ContinuousMaterial *_material;
    FEM_Shape *_shape;
    std::vector<std::vector<Matrix3x3> > JX_inv; // per element
    std::vector<std::vector<scalar> > V; // per element

    Grid_Level p1;
};