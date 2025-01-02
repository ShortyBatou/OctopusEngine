#pragma once
#include <array>
#include <random>
#include <Dynamic/FEM/FEM_Generic.h>

#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include <Dynamic/VBD/MG_VertexBlockDescent.h>
#include <Manager/Debug.h>

std::pair<int, int> P1_to_P2::ref_edges[6] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};

P1_to_P2::P1_to_P2(const Mesh::Topology& topology)
{
    std::set<int> visited;
    const int nb_P1 = elem_nb_vertices(Tetra);
    const int nb_P2 = elem_nb_vertices(Tetra10);
    for (int i = 0; i < topology.size(); i += nb_P2)
    {
        // for each particle on edges (red_id in [nb_p1, nb_p2])
        for (int j = nb_P1; j < nb_P2; ++j)
        {
            int vid = topology[i + j];
            if (visited.find(vid) == visited.end())
            {
                ids.push_back(topology[i + j]);
                const int a = ref_edges[j - nb_P1].first, b = ref_edges[j - nb_P1].second;
                edges.push_back(topology[i + a]);
                edges.push_back(topology[i + b]);
                visited.insert(vid);
            }
        }
    }
}

void P1_to_P2::prolongation(ParticleSystem* ps)
{
    for (int i = 0; i < ids.size(); i++)
    {
        const int a = edges[i*2], b = edges[i*2+1];
        Vector3 dx_a = ps->get(a)->position - ps->get(a)->last_position;
        Vector3 dx_b = ps->get(b)->position - ps->get(b)->last_position;
        ps->get(ids[i])->position = ps->get(ids[i])->last_position + (dx_a + dx_b) * 0.5f;
    }
}

int Q1_to_Q2::ref_edges[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 0}
};
int Q1_to_Q2::ref_faces[6][4] = {{0, 1, 2, 3}, {0, 1, 4, 5}, {1, 2, 5, 6}, {2, 3, 6, 7}, {3, 0, 7, 4}, {4, 5, 6, 7}};

void Q1_to_Q2::prolongation(ParticleSystem* ps)
{
    for (int i = 0; i < ids_edges.size(); i++)
    {
        const int a = edges[i*2], b = edges[i*2+1];
        Vector3 dx_a = ps->get(a)->position - ps->get(a)->last_position;
        Vector3 dx_b = ps->get(b)->position - ps->get(b)->last_position;
        ps->get(ids_edges[i])->position = ps->get(ids_edges[i])->last_position + (dx_a + dx_b) * 0.5f;
    }

    for (int i = 0; i < ids_faces.size(); i++)
    {
        Vector3 dx_sum = Unit3D::Zero();
        for (int j = 0; j < 4; j++)
        {
            dx_sum += ps->get(faces[i*4+j])->position - ps->get(faces[i*4+j])->last_position;
        }
        ps->get(ids_faces[i])->position = ps->get(ids_faces[i])->last_position + dx_sum * 0.25f;
    }

    for (int i = 0; i < ids_volumes.size(); i++)
    {
        Vector3 dx_sum = Unit3D::Zero();
        for (int j = 0; j < 8; j++)
        {
            dx_sum += ps->get(volume[i*8+j])->position - ps->get(volume[i*8+j])->last_position;
        }
        ps->get(ids_volumes[i])->position = ps->get(ids_volumes[i])->last_position + dx_sum * 0.125f;
    }
}


Q1_to_Q2::Q1_to_Q2(const Mesh::Topology& topology)
{
    std::set<int> visited;
    const int nb_Q1 = elem_nb_vertices(Hexa);
    const int nb_Q2 = elem_nb_vertices(Hexa27);
    for (int i = 0; i < topology.size(); i += nb_Q2)
    {
        int n = 0;
        // edges
        for (const auto& ref_edge : ref_edges)
        {
            const int vid = topology[i + nb_Q1 + n];
            if (visited.find(vid) == visited.end())
            {
                ids_edges.push_back(vid);
                const int a = ref_edge[0], b = ref_edge[1];
                edges.push_back(topology[i + a]);
                edges.push_back(topology[i + b]);
                visited.insert(vid);
            }
            n++;
        }

        // faces
        for (const auto& ref_face : ref_faces)
        {
            const int vid = topology[i + nb_Q1 + n];
            if (visited.find(vid) == visited.end())
            {
                ids_faces.push_back(vid);
                const int a = ref_face[0], b = ref_face[1];
                const int c = ref_face[2], d = ref_face[3];
                faces.push_back(topology[i + a]);
                faces.push_back(topology[i + b]);
                faces.push_back(topology[i + c]);
                faces.push_back(topology[i + d]);
                visited.insert(vid);
            }
            n++;
        }

        ids_volumes.push_back(topology[i + nb_Q1 + n]);
        for (int j = 0; j < 8; j++)
        {
            volume.push_back(topology[i + j]);
        }
    }
}

MG_VBD_FEM::MG_VBD_FEM(const Mesh::Topology& topology, const Mesh::Geometry& geometry, const Element e,
                       FEM_ContinuousMaterial* material, const scalar damp, const scalar density,
                       const Mass_Distribution distrib, const scalar linear, const int nb_iteration)
: VBD_FEM(topology, geometry, e, material, damp)
{
    // init global data and shape (P2)
    int it_linear = static_cast<int>(static_cast<scalar>(nb_iteration) * linear);
    int it_quad = nb_iteration - it_linear;
    _max_it = std::vector<int>({it_quad, it_linear});
    _it_count = 0;
    _current_grid = 1;

    _levels.push_back(data);
    _masses.push_back(compute_fem_mass(e, geometry, topology, density, distrib));

    const int nb_elem = static_cast<int>(topology.size()) / elem_nb_vertices(e);
    if (e == Tetra10)
    {
        // init constant for P1 => init grid[1]
        std::vector<int> linear_topo(nb_elem * 4);
        for (int i = 0; i < nb_elem; ++i)
            for (int j = 0; j < 4; ++j)
                linear_topo[i * 4 + j] = topology[i * 10 + j];

        _levels.push_back(build_data(linear_topo, geometry, Tetra));
        _masses.push_back(compute_fem_mass(Tetra, geometry, linear_topo, density, distrib));
        _masses.back().resize(_levels.back()->ids.size());
        // init prolongation
        _interpolation = new P1_to_P2(topology);
    }
    if (e == Hexa27)
    {
        std::vector<int> linear_topo(nb_elem * 8);
        for (int i = 0; i < nb_elem; ++i)
            for (int j = 0; j < 8; ++j)
                linear_topo[i * 8 + j] = topology[i * 27 + j];
        _levels.push_back(build_data(linear_topo, geometry, Hexa));
        _masses.push_back(compute_fem_mass(Hexa, geometry, linear_topo, distrib));
        _masses.back().resize(_levels.back()->ids.size());
        // init prolongation
        _interpolation = new Q1_to_Q2(topology);
    }
}


void MG_VBD_FEM::solve(VertexBlockDescent* ps, const scalar dt)
{

    _it_count++;
    const int last_grid = _current_grid;
    while (_it_count > _max_it[_current_grid])
    {
        _it_count = 1;
        _current_grid = (_current_grid + 1) % static_cast<int>(_levels.size());
    }

    if (last_grid != _current_grid)
    {
        interpolate(ps);
    }

    data = _levels[_current_grid];

    //std::shuffle(grid->_ids.begin(), grid->_ids.end(), std::mt19937());
    for (const int id : data->ids)
    {
        if (ps->get(id)->active) solve_vertex(ps, dt, id, _masses[_current_grid][id]);
    }

}

void MG_VBD_FEM::interpolate(VertexBlockDescent* ps) const
{
    // prolongatation
    if (_current_grid == 1)
    {
        _interpolation->prolongation(ps);
    }
}

void MG_VertexBlockDescent::step(const scalar dt)
{
    const scalar sub_dt = dt / static_cast<scalar>(_sub_iteration);
    for (int i = 0; i < _sub_iteration; ++i)
    {
        for (VBD_Object* obj : _objs) obj->compute_inertia(this, sub_dt);
        // get the first guess
        step_solver(sub_dt);
        scalar omega = 0;
        int ch_it = 0;
        for (int j = 0; j < _iteration; ++j)
        {
            for (VBD_Object* obj : _objs) obj->solve(this, sub_dt);
            step_effects(sub_dt);
            step_constraint(sub_dt);
            for (const MG_VBD_FEM* obj : _fems) obj->interpolate(this);
            //chebyshev_acceleration(j, omega);
            //for(MG_VBD_FEM* obj : _fems) obj->interpolate(this, sub_dt);
            ch_it++;
        }

        //for (MG_VBD_FEM* obj : _fems) obj->interpolate(this);
        update_velocity(sub_dt);
    }
    reset_external_forces();
}
