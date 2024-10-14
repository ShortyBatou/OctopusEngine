#pragma once
#include <random>
#include <set>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"

struct Grid_Level {
    Grid_Level(FEM_Shape *shape, const Mesh::Geometry &positions, const Mesh::Topology &topology, const std::vector<scalar> &masses,
        const std::vector<std::vector<Matrix3x3>> &jx_inv, const std::vector<std::vector<scalar>> &v, const Mesh::Geometry &y)
        : _shape(shape),
          _masses(masses),
          _JX_inv(jx_inv),
          _V(v),
          _positions(positions),
          _y(y) {
    }

    // const
    FEM_Shape *_shape; 
    std::vector<scalar> _masses; // for each vertice
    std::vector<std::vector<Matrix3x3>> _JX_inv; // per element
    std::vector<std::vector<scalar>> _V; // per element
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice

    // dynamic
    Mesh::Geometry _positions; // each iteration
    Mesh::Geometry _y; // each integration
};

struct Grid_Interpolation {
    virtual ~Grid_Interpolation() = default;
    virtual void restriction(const Grid_Level* gH, Grid_Level* g2H) = 0;
    virtual void prolongation(const Grid_Level* g2H, Grid_Level* gH) = 0;
};

struct MG_VBD_FEM {
    MG_VBD_FEM(const Mesh::Topology &topology, ParticleSystem* ps, FEM_Shape *shape, FEM_ContinuousMaterial *material) {

    }

    void build_fem_const(const Mesh::Topology &topology,  ParticleSystem* ps, FEM_Shape *shape) {

    }


    void build_neighboors(const Mesh::Topology &topology);



    void solve(ParticleSystem *ps, scalar dt);

    void plot_residual(ParticleSystem *ps, scalar dt);

    scalar compute_energy(ParticleSystem *ps) const;

    std::vector<Vector3> compute_forces(ParticleSystem *ps, scalar dt) const;

    void solve_vertex(ParticleSystem *ps, scalar dt, int vid);

    void solve_element(ParticleSystem *ps, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i);

    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);

    void compute_inertia(ParticleSystem *ps, scalar dt);

protected:

    FEM_ContinuousMaterial *_material;
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice
    std::vector<Grid_Level*> grids;
};