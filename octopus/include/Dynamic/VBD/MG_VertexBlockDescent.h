#pragma once
#include <random>
#include <set>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/VBD/VBD_Object.h"

struct Grid_Level {
    Grid_Level(FEM_Shape *shape,
               const std::vector<scalar> &masses,
               const std::vector<std::vector<Matrix3x3> > &jx_inv,
               const std::vector<std::vector<scalar> > &v, std::vector<int> ids) : _shape(shape), _masses(masses),
        _JX_inv(jx_inv), _V(v), _ids(ids) {
    }

    // const
    FEM_Shape *_shape; // element used in this grid level
    std::vector<int> _ids; // vertices that are used in this grid level
    std::vector<scalar> _masses; // for each vertice
    std::vector<std::vector<Matrix3x3> > _JX_inv; // per element
    std::vector<std::vector<scalar> > _V; // per element
};

struct GridInterpolation {
    virtual ~GridInterpolation() = default;
    virtual void prolongation(ParticleSystem *ps, std::vector<Vector3> dx) = 0;
};

struct P1_to_P2 final : GridInterpolation {
    std::vector<int> ids;
    std::vector<std::pair<int, int> > edges;
    static std::pair<int, int> ref_edges[6];

    explicit P1_to_P2(const Mesh::Topology &topology);

    void prolongation(ParticleSystem *ps, std::vector<Vector3> dx) override;

    ~P1_to_P2() override = default;
};

struct Q1_to_Q2 final : GridInterpolation {
    std::vector<int> ids_edges;
    std::vector<int> ids_faces;
    std::vector<int> ids_volumes;
    std::vector<std::array<int, 2>> edges;
    std::vector<std::array<int, 4>> faces;
    std::vector<std::array<int, 8>> volume;

    static int ref_edges[12][2];
    static int ref_faces[6][4];

    explicit Q1_to_Q2(const Mesh::Topology &topology);

    void prolongation(ParticleSystem *ps, std::vector<Vector3> dx) override;

    ~Q1_to_Q2() override = default;
};

struct MG_VBD_FEM final : VBD_Object {
    MG_VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, Element e,
               FEM_ContinuousMaterial *material, scalar damp, scalar density);

    void build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry, scalar density, Element e);
    void build_neighboors(const Mesh::Topology &topology);


    void solve(ParticleSystem *ps, scalar dt);

    void plot_residual(ParticleSystem *ps, Grid_Level* grid, scalar dt) ;
    scalar compute_energy(ParticleSystem *ps, Grid_Level* grid) const;
    std::vector<Vector3> compute_forces(ParticleSystem *ps, Grid_Level* grid, scalar dt) const ;

    // int i = vertex index for ids
    void solve_vertex(ParticleSystem *ps, Grid_Level *grid, scalar dt, int i);
    void solve_element(ParticleSystem *ps, const Grid_Level *grid, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i);
    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);
    void compute_inertia(ParticleSystem *ps, scalar dt) override;

protected:
    scalar _k_damp;
    std::vector<Grid_Level *> _grids;
    GridInterpolation *_interpolation;
    FEM_ContinuousMaterial *_material;
    std::vector<Vector3> _y;
    std::vector<Vector3> _dx;
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice
    FEM_Shape *_shape;
    Mesh::Topology _topology;
};
