#pragma once
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
    for (int i = 0; i < topology.size(); i += 10) {
        for (int j = 0; j < 6; ++j) {
            int vid = topology[i + 4 + j];
            if(visited.find(vid) == visited.end()) {
                ids.push_back(topology[i + 4 + j]);
                const int a = ref_edges[j].first, b = ref_edges[j].second;
                edges.emplace_back(topology[i + a], topology[i + b]);
                visited.insert(vid);
            }

        }
    }
}

void P1_to_P2::prolongation(ParticleSystem *ps, std::vector<Vector3> dx) {
    for (int i = 0; i < ids.size(); i++) {
        const int a = edges[i].first, b = edges[i].second;
        ps->get(ids[i])->position += (dx[a] + dx[b]) * 0.5f;
    }
}


MG_VBD_FEM::MG_VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape *shape,
               FEM_ContinuousMaterial *material, scalar damp, scalar density) {
    // init global data and shape (P2)
    _shape = shape;
    _y = geometry;
    _k_damp = damp;
    _material = material;
    _owners.resize(geometry.size());
    _ref_id.resize(geometry.size());
    _dx.resize(geometry.size());
    _topology = topology;
    // init neighboors for each particles (same for each level)
    build_neighboors(topology);
    // init constant for P2 => init grid[0]
    build_fem_const(topology, geometry, density, Tetra10);
    // init constant for P1 => init grid[1]
    build_fem_const(topology, geometry, density, Tetra);

    // init prolongation
    p1_to_p2 = new P1_to_P2(topology);
}

void MG_VBD_FEM::build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry, scalar density, Element e) {
    FEM_Shape *l_shape = get_fem_shape(e);
    const int nb_quadrature = l_shape->nb_quadratures();
    const int nb_element = static_cast<int>(topology.size()) / _shape->nb;
    std::vector<std::vector<Matrix3x3> > JX_inv(nb_element);
    std::vector<std::vector<scalar> > V(nb_element);
    std::vector<scalar> masses(geometry.size(), 0);
    std::set<int> s_ids;
    for (int i = 0; i < nb_element; i++) {
        scalar mass = 0;
        const int id = i * _shape->nb;

        V[i].resize(nb_quadrature);
        JX_inv[i].resize(nb_quadrature);

        for (int j = 0; j < nb_quadrature; ++j) {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < l_shape->nb; ++k) {
                s_ids.insert(topology[id + k]); // save all ids of this level
                J += glm::outerProduct(geometry[topology[id + k]], l_shape->dN[j][k]);
            }
            V[i][j] = abs(glm::determinant(J)) * l_shape->weights[j];
            JX_inv[i][j] = glm::inverse(J);
            mass += V[i][j];
        }
        mass *= density / static_cast<scalar>(l_shape->nb);
        for (int k = 0; k < _shape->nb; ++k) {
            masses[topology[id + k]] = mass;
        }
    }
    const std::vector ids(s_ids.begin(), s_ids.end());
    grids.push_back(new Grid_Level(l_shape, masses, JX_inv, V, ids));
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
    int it1 = 1, it2 = 4;
    Grid_Level *grid;

    grid = grids[1];
    std::fill(_dx.begin(), _dx.end(), Unit3D::Zero());
    for(int i = 0; i < it1; ++i) {
        for (const int id: grid->_ids) {
            if(ps->get(id)->active) solve_vertex(ps, grid, dt, id);
        }
    }

    // prolongatation
    p1_to_p2->prolongation(ps, _dx);
    grid = grids[0];
    for(int i = 0; i < it2; ++i) {
        for (const int id: grid->_ids) {
            if(ps->get(id)->active) solve_vertex(ps, grid, dt, id);
        }
    }
    plot_residual(ps, grids[0], dt);
}

void MG_VBD_FEM::plot_residual(ParticleSystem *ps, Grid_Level* grid,  scalar dt) {
    const int nb_vertices = static_cast<int>(_owners.size());
    const scalar e = compute_energy(ps, grid);
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
    DebugUI::Begin("MG Forces");
    DebugUI::Value("MG Forces val", sum);
    DebugUI::Plot("MG Forces norm", sum, 200);
    DebugUI::Range("MG Forces range", sum);
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

// int i = vertex index for ids
void MG_VBD_FEM::solve_vertex(ParticleSystem *ps, Grid_Level *grid, scalar dt, int i) {
    const int vid = grid->_ids[i];
    // need current grid info
    const int nb_owners = static_cast<int>(_owners[vid].size());
    Vector3 f_i = Unit3D::Zero();
    Matrix3x3 H_i = Matrix::Zero3x3();

    for (int j = 0; j < nb_owners; ++j) {
        const int owner = _owners[vid][j];
        const int ref_id = _ref_id[vid][j];
        solve_element(ps, grid, owner, ref_id, f_i, H_i);
    }
    // Inertia
    Particle *p = ps->get(vid);
    f_i += -p->mass / (dt * dt) * (p->position - _y[vid]);
    H_i += Matrix3x3(p->mass / (dt * dt));

    f_i -= _k_damp / dt * H_i * (p->position - p->last_position);
    H_i += _k_damp / dt * H_i;

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
            Vector3 p = ps->get(vid)->position;
            Jx += glm::outerProduct(p, grid->_shape->dN[i][j]);
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
        _y[i] = p->position + p->velocity * dt + (
                    (p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt * dt;
    }
}