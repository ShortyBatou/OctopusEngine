#pragma once
#include <random>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/VBD/VBD_Object.h"
#include "Dynamic/VBD/VertexBlockDescent.h"

struct VBD_FEM_Data {
    VBD_FEM_Data(FEM_Shape *_shape, const std::vector<int>& _topo,
               const std::vector<std::vector<Matrix3x3> > &_jx_inv,
               const std::vector<std::vector<scalar> > &_v, const std::vector<int>& _ids) :
        shape(_shape), topology(_topo), JX_inv(_jx_inv), V(_v), ids(_ids) {
    }

    // const
    FEM_Shape *shape; // element used in this grid level
    Mesh::Topology topology;
    std::vector<int> ids; // vertices that are used in this grid level
    std::vector<std::vector<Matrix3x3> > JX_inv; // per element
    std::vector<std::vector<scalar> > V; // per element
};

struct VBD_FEM : VBD_Object {
    VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, Element e,
            FEM_ContinuousMaterial *material, scalar k_damp = 0.f);

    void compute_inertia(VertexBlockDescent* vbd, scalar dt) override;

    void build_neighboors(const Mesh::Topology &topology);

    VBD_FEM_Data* build_data(const Mesh::Topology &topology, const Mesh::Geometry &geometry, Element e);

    void solve(VertexBlockDescent *vbd, scalar dt) override;

    void plot_residual(VertexBlockDescent *vbd, scalar dt) const;

    std::vector<scalar> compute_stress(VertexBlockDescent *vbd) const;
    std::vector<scalar> compute_volume(VertexBlockDescent *vbd) const;
    std::vector<scalar> compute_colume_diff(VertexBlockDescent *vbd) const;
    std::vector<Vector3> compute_forces(VertexBlockDescent *vbd, scalar dt) const;

    virtual void solve_vertex(VertexBlockDescent *vbd, scalar dt, int vid, scalar mass);

    virtual void solve_element(VertexBlockDescent *vbd, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i);

    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);

    ~VBD_FEM() override
    {
        delete _material;
        delete data;
    }

    scalar _k_damp;
    std::vector<std::vector<int>> _owners; // for each vertice
    std::vector<std::vector<int>> _ref_id; // for each vertice
    std::vector<Vector3> _y; // inertia
    FEM_ContinuousMaterial *_material;
    VBD_FEM_Data* data;
};
