#pragma once
#include <array>
#include <random>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include <Dynamic/VBD/MG_VertexBlockDescent.h>
#include <Manager/Debug.h>
#include <Manager/Input.h>
std::pair<int, int> P1_to_P2::ref_edges[6] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};

P1_to_P2::P1_to_P2(const Mesh::Topology &topology) {
    std::set<int> visited;
    const int nb_P1 = elem_nb_vertices(Tetra);
    const int nb_P2 = elem_nb_vertices(Tetra10);
    const int diff = nb_P2 - nb_P1;
    for (int i = 0; i < topology.size(); i += nb_P2) {
        // for each particle on edges (red_id in [nb_p1, nb_p2])
        for (int j = nb_P1; j < nb_P2; ++j) {
            int vid = topology[i + j];
            if(visited.find(vid) == visited.end()) {
                ids.push_back(topology[i + j]);
                const int a = ref_edges[j-nb_P1].first, b = ref_edges[j-nb_P1].second;
                edges.emplace_back(topology[i + a], topology[i + b]);
                visited.insert(vid);
            }
        }
    }
}

void P1_to_P2::prolongation(ParticleSystem *ps, const std::vector<Vector3>& y, const std::vector<Vector3>& dx) {
    for (int i = 0; i < ids.size(); i++) {
        if(ps->get(ids[i])->active) {
            const int a = edges[i].first, b = edges[i].second;
            // place holder that take in count the "correction" of constrained particles to counter inertia
            if(!ps->get(a)->active) {
                ps->get(ids[i])->position = y[ids[i]] + ((ps->get(a)->position - y[a]) + dx[b]) * 0.5f;
            }
            else if(!ps->get(b)->active) {
                ps->get(ids[i])->position = y[ids[i]] + ((ps->get(b)->position - y[b]) + dx[a]) * 0.5f;
            }
            else {
                ps->get(ids[i])->position = y[ids[i]] + (dx[a] + dx[b]) * 0.5f;
            }

        }
    }
}

P1_to_P2_Mass::P1_to_P2_Mass(const Mesh::Topology &topology) {
    int max_id = *std::max_element(topology.begin(), topology.end()) + 1;
    ids.resize(max_id);
    weights.resize(max_id);
    const int nb_P1 = elem_nb_vertices(Tetra);
    const int nb_P2 = elem_nb_vertices(Tetra10);
    const int diff = nb_P2 - nb_P1;

    std::vector<std::vector<int>> owners(max_id);
    std::vector<std::vector<int>> ref_id(max_id);
    for (int i = 0; i < topology.size(); i += nb_P2) {
        int eid = i / nb_P2;
        for(int j = 0; j < nb_P2; ++j) {
            const int vid = topology[i+j];
            owners[vid].push_back(eid);
            ref_id[vid].push_back(j);
        }
    }

    // for each point
    for(int i = 0; i < owners.size(); ++i) {
        // we get the weights
        for(int j = 0; j < owners[i].size(); ++j) {
            const int eid = owners[i][j];
            const int rid = ref_id[i][j];
            if(rid < 4) {
                for(int k = 0; k < 4; ++k) {
                    ids[i].push_back(topology[eid * nb_P2 + k]);
                    if(k == rid) weights[i].push_back(0.2);
                    else weights[i].push_back(-0.1333333);
                }
            }
            else {
                std::pair<int,int> edge = P1_to_P2::ref_edges[rid-4];
                for(int k = 0; k < 4; ++k) {
                    ids[i].push_back(topology[eid * nb_P2 + k]);
                    if(rid == edge.first || rid == edge.second) weights[i].push_back(0.53333);
                    else weights[i].push_back(-0.1333333);
                }
            }
        }
    }
}

void P1_to_P2_Mass::prolongation(ParticleSystem *ps, const std::vector<Vector3>& y, const std::vector<Vector3>& dx) {
    for (int i = 0; i < ps->nb_particles(); i++) {
       ps->get(i)->position = ps->get(i)->last_position;
    }
    for (int i = 0; i < ids.size(); i++) {
        for(int j = 0; j < ids[i].size(); ++j) {
            ps->get(i)->position += dx[ids[i][j]] * weights[i][j];
        }
    }
}


int Q1_to_Q2::ref_edges[12][2] = {{0, 1},{1, 2},{2, 3},{3, 0},{0, 4},{1, 5},{2, 6},{3, 7},{4, 5},{5, 6},{6, 7},{7, 0}};
int Q1_to_Q2::ref_faces[6][4] = {{0,1,2,3},{0,1,4,5},{1,2,5,6},{2,3,6,7},{3,0,7,4},{4,5,6,7}};

void Q1_to_Q2::prolongation(ParticleSystem *ps, const std::vector<Vector3>& y, const std::vector<Vector3>& dx) {
    for (int i = 0; i < ids_edges.size(); i++) {
        const int a = edges[i][0], b = edges[i][1];
        ps->get(ids_edges[i])->position = y[ids_edges[i]] + (dx[a] + dx[b]) * 0.5f;
    }

    for (int i = 0; i < ids_faces.size(); i++) {
        Vector3 dx_sum = Unit3D::Zero();
        for(int j = 0; j < 4; j++) {
            dx_sum += dx[faces[i][j]];
        }
        ps->get(ids_faces[i])->position = y[ids_faces[i]] + dx_sum * 0.25f;
    }

    for (int i = 0; i < ids_volumes.size(); i++) {
        Vector3 dx_sum = Unit3D::Zero();
        for(int j = 0; j < 8; j++) {
            dx_sum += dx[volume[i][j]];
        }
        ps->get(ids_volumes[i])->position = y[ids_volumes[i]] + dx_sum * 0.125f;
    }
}


Q1_to_Q2::Q1_to_Q2(const Mesh::Topology &topology) {
    std::set<int> visited;
    const int nb_Q1 = elem_nb_vertices(Hexa);
    const int nb_Q2 = elem_nb_vertices(Hexa27);
    for (int i = 0; i < topology.size(); i += nb_Q2) {
        int n = 0;
        // edges
        for (const auto & ref_edge : ref_edges) {
            const int vid = topology[i + nb_Q1 + n];
            if(visited.find(vid) == visited.end()) {
                ids_edges.push_back(vid);
                const int a = ref_edge[0], b = ref_edge[1];
                edges.push_back({topology[i + a], topology[i + b]});
                visited.insert(vid);
            }
            n++;
        }

        // faces
        for (const auto & ref_face : ref_faces) {
            const int vid = topology[i + nb_Q1 + n];
            if(visited.find(vid) == visited.end()) {
                ids_faces.push_back(vid);
                const int a = ref_face[0], b = ref_face[1];
                const int c = ref_face[2], d = ref_face[3];
                faces.push_back({topology[i + a], topology[i + b],topology[i + c],topology[i + d]});
                visited.insert(vid);
            }
            n++;
        }

        ids_volumes.push_back(topology[i + nb_Q1 + n]);
        std::array<int,8> ids{};
        for(int j = 0; j < 8; j++) {
            ids[j] = topology[i + j];
        }
        volume.push_back(ids);
    }
}


MG_VBD_FEM::MG_VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, Element e,
               FEM_ContinuousMaterial *material, scalar damp, scalar density) {
    // init global data and shape (P2)
    _shape = get_fem_shape(e);
    _y = geometry;
    _k_damp = damp;
    _material = material;
    _owners.resize(geometry.size());
    _ref_id.resize(geometry.size());
    _dx.resize(geometry.size());
    _topology = topology;
    // init neighboors for each particles (same for each level)
    build_neighboors(topology);
    if(e == Tetra10) {
        // init constant for P2 => init grid[0]
        build_fem_const(topology, geometry, density, Tetra10);
        // init constant for P1 => init grid[1]
        build_fem_const(topology, geometry, density, Tetra);

        // init prolongation
        _interpolation = new P1_to_P2(topology);
    }
    if(e == Hexa27) {
        // init constant for P2 => init grid[0]
        build_fem_const(topology, geometry, density, Hexa27);
        // init constant for P1 => init grid[1]
        build_fem_const(topology, geometry, density, Hexa);

        // init prolongation
        _interpolation = new Q1_to_Q2(topology);
    }
}

void MG_VBD_FEM::build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry, scalar density, Element e) {
    FEM_Shape *l_shape = get_fem_shape(e);
    const int nb_quadrature = l_shape->nb_quadratures();
    const int nb_element = static_cast<int>(topology.size()) / _shape->nb;
    std::vector<std::vector<Matrix3x3> > JX_inv(nb_element);
    std::vector<std::vector<scalar> > V(nb_element);
    std::vector<scalar> masses(geometry.size(), 0.f);
    std::set<int> s_ids;
    for (int i = 0; i < nb_element; i++) {
        const int id = i * _shape->nb; // first vertex of element

        V[i].resize(nb_quadrature);
        JX_inv[i].resize(nb_quadrature);
        scalar V_elem = 0;
        for (int j = 0; j < nb_quadrature; ++j) {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < l_shape->nb; ++k) {
                s_ids.insert(topology[id + k]); // save all ids of this level
                J += glm::outerProduct(geometry[topology[id + k]], l_shape->dN[j][k]);
            }
            V[i][j] = abs(glm::determinant(J)) * l_shape->weights[j];
            JX_inv[i][j] = glm::inverse(J);
            V_elem += V[i][j];
        }
        for (int k = 0; k < l_shape->nb; ++k) {
            masses[topology[id] + k] += V_elem * density / l_shape->nb;
        }
    }

    const std::vector ids(s_ids.begin(), s_ids.end());
    _grids.push_back(new Grid_Level(l_shape, masses, JX_inv, V, ids));
}


void MG_VBD_FEM::build_neighboors(const Mesh::Topology &topology) {
    for (int i = 0; i < topology.size(); i += _shape->nb) {
        for (int j = 0; j < _shape->nb; ++j) {
            _owners[topology[i + j]].push_back(i / _shape->nb);
            _ref_id[topology[i + j]].push_back(j);
        }
    }
}


void MG_VBD_FEM::solve(ParticleSystem *ps, scalar dt) {
    // coarse to refined (P1=>P2)
    int it1 = 1, it2 = 0;
    Grid_Level *grid;

    plot_residual(ps, _grids[0], dt, 0);
    grid = _grids[1];
    std::fill(_dx.begin(), _dx.end(), Unit3D::Zero());
    for(int i = 0; i < it1; ++i) {
        std::iota(grid->_ids.begin(), grid->_ids.end(), 0);
        std::shuffle(grid->_ids.begin(), grid->_ids.end(), std::mt19937());
        for (const int id: grid->_ids) {
            if(ps->get(id)->active)
                solve_vertex(ps, grid, dt, id);
        }
        plot_residual(ps, _grids[0], dt, 0);
    }

    // prolongatation
    _interpolation->prolongation(ps, _y, _dx);
    plot_residual(ps, _grids[0], dt, 0);

    grid = _grids[0];
    for(int i = 0; i < it2; ++i) {
        std::iota(grid->_ids.begin(), grid->_ids.end(), 0);
        std::shuffle(grid->_ids.begin(), grid->_ids.end(), std::mt19937());
        for (const int id: grid->_ids) {
            if(ps->get(id)->active) solve_vertex(ps, grid, dt, id);
        }
        plot_residual(ps, _grids[0], dt, 0);
    }
}

void MG_VBD_FEM::plot_residual(ParticleSystem *ps, Grid_Level* grid,  scalar dt, int id = 0) {
    const int nb_vertices = static_cast<int>(_owners.size());
    const std::vector<Vector3> forces = compute_forces(ps, grid, dt);
    scalar sum = 0;
    scalar total = 0;
    for (int i = 0; i < nb_vertices; ++i) {
        const Particle *p = ps->get(i);
        if (p->active) {
            total++;
            sum += glm::dot(forces[i], forces[i]);
        }
    }
    sum /= total;
    DebugUI::Begin(std::to_string(id) + " MG Forces ");
    DebugUI::Value(std::to_string(id) + " MG Forces val ", sum);
    DebugUI::Plot(std::to_string(id) + " MG Forces norm", sum, 200);
    DebugUI::Range(std::to_string(id) + " MG Forces range", sum);
    DebugUI::End();
}

scalar MG_VBD_FEM::compute_energy(ParticleSystem *ps, Grid_Level* grid) const {
    // can wait
    return 0;
}

std::vector<Vector3> MG_VBD_FEM::compute_forces(ParticleSystem *ps, Grid_Level* grid, scalar dt) const {
    std::vector forces(ps->nb_particles(), Unit3D::Zero());
    const int &nb_vert_elem = _shape->nb;
    const int nb_quadrature = _shape->nb_quadratures();
    for (int e = 0; e < _topology.size(); e += _shape->nb) {
        const int eid = e / _shape->nb;
        for (int i = 0; i < nb_quadrature; ++i) {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = _topology[eid * nb_vert_elem + j];
                Jx += glm::outerProduct(ps->get(vid)->position, _shape->dN[i][j]);
            }

            Matrix3x3 F = Jx * grid->_JX_inv[eid][i];
            Matrix3x3 P = _material->get_pk1(F) * glm::transpose(grid->_JX_inv[eid][i]) * grid->_V[eid][i];
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = _topology[eid * nb_vert_elem + j];
                forces[vid] -= P * _shape->dN[i][j];
            }
        }
    }
    for (int i = 0; i < ps->nb_particles(); ++i) {
        const Particle *p = ps->get(i);
        forces[i] += -p->mass / (dt * dt) * (p->position - _y[i]);
    }
    return forces;
}

void MG_VBD_FEM::solve_vertex(ParticleSystem *ps, Grid_Level *grid, scalar dt, int vid) {
    const int nb_owners = static_cast<int>(_owners[vid].size());
    Vector3 f_i = Unit3D::Zero();
    Matrix3x3 H_i = Matrix::Zero3x3();
    scalar mass = grid->_masses[vid];
    Particle *p = ps->get(vid);
    for (int j = 0; j < nb_owners; ++j) {
        const int owner = _owners[vid][j];
        const int ref_id = _ref_id[vid][j];
        solve_element(ps, grid, owner, ref_id, f_i, H_i);
    }

    // Damping
    f_i -= _k_damp / dt * H_i * (p->position - p->last_position);
    H_i += _k_damp / dt * H_i;

    // Inertia
    f_i += -mass / (dt * dt) * (p->position - _y[vid]);
    H_i += Matrix3x3(mass / (dt * dt));

    const scalar detH = abs(glm::determinant(H_i));
    const Vector3 dx = detH > eps ? glm::inverse(H_i) * f_i : Unit3D::Zero();
    p->position += dx;
    _dx[vid] += dx;
}

void MG_VBD_FEM::solve_element(ParticleSystem *ps, const Grid_Level *grid, const int eid, const int ref_id, Vector3 &f_i, Matrix3x3 &H_i) {
    const int nb_quadrature = grid->_shape->nb_quadratures();
    const int nb_vert_elem = grid->_shape->nb;
    const int nb_vert_elem_max = _shape->nb;
    std::vector<Matrix3x3> d2Psi_dF2(9);
    for (int i = 0; i < nb_quadrature; ++i) {
        Matrix3x3 Jx = Matrix::Zero3x3();
        for (int j = 0; j < nb_vert_elem; ++j) {
            const int vid = _topology[eid * nb_vert_elem_max + j];
            Jx += glm::outerProduct(ps->get(vid)->position, grid->_shape->dN[i][j]);
        }

        Matrix3x3 F = Jx * grid->_JX_inv[eid][i];
        Vector3 dF_dx = glm::transpose(grid->_JX_inv[eid][i]) * grid->_shape->dN[i][ref_id];

        // compute force
        Matrix3x3 P = _material->get_pk1(F);
        f_i -= P * dF_dx * grid->_V[eid][i];

        _material->get_sub_hessian(F, d2Psi_dF2);
        H_i += assemble_hessian(d2Psi_dF2, dF_dx) * grid->_V[eid][i];
    }
}

Matrix3x3 MG_VBD_FEM::assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, const Vector3 dF_dx) {
    Matrix3x3 H = Matrix::Zero3x3();
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            Matrix3x3 H_kl;
            for (int l = 0; l < 3; ++l) {
                for (int k = 0; k < 3; ++k) {
                    H_kl[k][l] = d2W_dF2[k + l * 3][i][j];
                }
            }
            H[i][j] = glm::dot(dF_dx, H_kl * dF_dx);
        }
    }
    return H;
}

void MG_VBD_FEM::compute_inertia(ParticleSystem *ps, scalar dt) {
    // normally we should make a better approximation but osef
    for (int i = 0; i < ps->nb_particles(); ++i) {
        const Particle *p = ps->get(i);
        _y[i] = p->position + p->velocity * dt + ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt * dt;
    }
}