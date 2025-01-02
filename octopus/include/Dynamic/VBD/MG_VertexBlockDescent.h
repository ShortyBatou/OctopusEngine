#pragma once
#include <random>
#include <set>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/VBD/VertexBlockDescent.h"
#include "Dynamic/VBD/VBD_FEM.h"


struct GridInterpolation
{
    virtual ~GridInterpolation() = default;
    virtual void prolongation(ParticleSystem* ps) = 0;
};

struct P1_to_P2 final : GridInterpolation
{
    std::vector<int> ids;
    std::vector<int> edges;
    static std::pair<int, int> ref_edges[6];

    explicit P1_to_P2(const Mesh::Topology& topology);

    void prolongation(ParticleSystem* ps) override;

    ~P1_to_P2() override = default;
};

struct Q1_to_Q2 final : GridInterpolation
{
    std::vector<int> ids_edges;
    std::vector<int> ids_faces;
    std::vector<int> ids_volumes;
    std::vector<int> edges;
    std::vector<int> faces;
    std::vector<int> volume;

    static int ref_edges[12][2];
    static int ref_faces[6][4];

    explicit Q1_to_Q2(const Mesh::Topology& topology);

    void prolongation(ParticleSystem* ps) override;

    ~Q1_to_Q2() override = default;
};


struct MG_VBD_FEM final : VBD_FEM
{
    MG_VBD_FEM(const Mesh::Topology& topology, const Mesh::Geometry& geometry, Element e,
               FEM_ContinuousMaterial* material, scalar damp, scalar density, Mass_Distribution distrib, scalar linear, int nb_iteration);

    void solve(VertexBlockDescent* ps, scalar dt) override;
    void interpolate(VertexBlockDescent* ps) const;
    bool check_level(VertexBlockDescent* ps);
    ~MG_VBD_FEM() override
    {
        delete _interpolation;
        // levels[0] is data and will be deleted by VBD_FEM destructor
        for (int i = 1; i < _levels.size(); ++i) delete _levels[i];
    }

protected:
    std::vector<int> _max_it;
    int _it_count;
    int _current_grid;
    std::vector<std::vector<scalar>> _masses;
    std::vector<VBD_FEM_Data*> _levels;
    GridInterpolation* _interpolation;
};

struct MG_VertexBlockDescent final : VertexBlockDescent
{
    explicit MG_VertexBlockDescent(Solver* solver, const int iteration, const int sub_iteration, const scalar rho)
        : VertexBlockDescent(solver, iteration, sub_iteration, rho)
    { }

    void step(scalar dt) override;

    void add_fem(MG_VBD_FEM* obj)
    {
        _objs.push_back(obj);
        _fems.push_back(obj);
    }

    ~MG_VertexBlockDescent() override
    {
        for (const MG_VBD_FEM* fem : _fems) delete fem;
    }

protected:
    std::vector<MG_VBD_FEM*> _fems;
};
