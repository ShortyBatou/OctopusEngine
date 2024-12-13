#pragma once
#include <random>
#include <set>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/VBD/VertexBlockDescent.h"

struct Grid_Level {
    Grid_Level(FEM_Shape *shape,
               const std::vector<scalar> &masses,
               const std::vector<std::vector<Matrix3x3> > &jx_inv,
               const std::vector<std::vector<scalar> > &v, const std::vector<int>& ids) : _shape(shape), _masses(masses),
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
    virtual void prolongation(ParticleSystem *ps, const std::vector<Vector3>& y) = 0;
};

struct P1_to_P2 final : GridInterpolation {
    std::vector<int> ids;
    std::vector<std::pair<int, int> > edges;
    static std::pair<int, int> ref_edges[6];

    explicit P1_to_P2(const Mesh::Topology &topology);

    void prolongation(ParticleSystem *ps, const std::vector<Vector3>& y) override;

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

    void prolongation(ParticleSystem *ps, const std::vector<Vector3>& y) override;

    ~Q1_to_Q2() override = default;
};


struct MG_VBD_FEM final : VBD_Object {
    MG_VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, Element e,
               FEM_ContinuousMaterial *material, scalar damp, scalar density);

    void build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry, scalar density, Element e);
    void build_neighboors(const Mesh::Topology &topology);


    void solve(VertexBlockDescent *ps, scalar dt) override;

    void interpolate(VertexBlockDescent *ps, scalar dt);

    void plot_residual(VertexBlockDescent *ps, Grid_Level* grid, scalar dt, int id) ;
    scalar compute_energy(VertexBlockDescent *ps, Grid_Level* grid) const;
    std::vector<Vector3> compute_forces(VertexBlockDescent *ps, const Grid_Level* grid, scalar dt) const ;

    // int i = vertex index for ids
    void solve_vertex(VertexBlockDescent *ps, Grid_Level *grid, scalar dt, int i);
    void solve_element(VertexBlockDescent *ps, const Grid_Level *grid, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i);
    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);
    void compute_inertia(VertexBlockDescent *ps, scalar dt) override;

protected:
    std::vector<int> _max_it;
    int _it_count;
    int _current_grid;
    scalar _k_damp;
    std::vector<Grid_Level *> _grids;
    GridInterpolation *_interpolation;
    FEM_ContinuousMaterial *_material;
    std::vector<Vector3> _y;
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice
    FEM_Shape *_shape;
    Mesh::Topology _topology;
};

struct MG_VertexBlockDescent final : VertexBlockDescent {
    explicit MG_VertexBlockDescent(Solver *solver, const int iteration, const int sub_iteration, const scalar rho)
        : VertexBlockDescent(solver, iteration, sub_iteration, rho) {
    }

    void step(scalar dt) override;
    void add_fem(MG_VBD_FEM *obj) {
        _objs.push_back(obj);
        _fems.push_back(obj);
    }

    ~MG_VertexBlockDescent() override {
    }
protected:
    std::vector<MG_VBD_FEM *> _fems;
};