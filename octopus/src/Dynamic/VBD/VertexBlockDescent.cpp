#pragma once
#include <random>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include <Dynamic/VBD/VertexBlockDescent.h>
#include <Manager/Debug.h>
#include <Manager/Input.h>


VBD_FEM::VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape *shape,
                 FEM_ContinuousMaterial *material, scalar k_damp) : _shape(shape), _material(material), _topology(topology), _k_damp(k_damp) {
    _owners.resize(geometry.size());
    _ref_id.resize(geometry.size());
    build_fem_const(topology, geometry);
    build_neighboors(topology);
    _y = geometry;
}

void VBD_FEM::build_neighboors(const Mesh::Topology &topology) {
    for (int i = 0; i < topology.size(); i += _shape->nb) {
        for (int j = 0; j < _shape->nb; ++j) {
            _owners[topology[i + j]].push_back(i / _shape->nb);
            _ref_id[topology[i + j]].push_back(j);
        }
    }
}

void VBD_FEM::build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry) {
    const int nb_quadrature = _shape->nb_quadratures();
    const int nb_element = static_cast<int>(topology.size()) / _shape->nb;

    JX_inv.resize(nb_element);
    V.resize(nb_element);

    for (int i = 0; i < nb_element; i++) {
        const int id = i * _shape->nb;
        V[i].resize(nb_quadrature);
        JX_inv[i].resize(nb_quadrature);
        for (int j = 0; j < nb_quadrature; ++j) {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < _shape->nb; ++k) {
                J += glm::outerProduct(geometry[topology[id + k]], _shape->dN[j][k]);
            }
            V[i][j] = abs(glm::determinant(J)) * _shape->weights[j];
            JX_inv[i][j] = glm::inverse(J);
        }
    }
}

void VBD_FEM::solve(ParticleSystem *ps, const scalar dt) {
    const int nb_vertices = static_cast<int>(_owners.size());
    std::vector<int> ids(nb_vertices);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), std::mt19937());
    for (int i = 0; i < nb_vertices; ++i) {
        if (ps->get(ids[i])->active) solve_vertex(ps, dt, ids[i]);
    }
}

void VBD_FEM::plot_residual(ParticleSystem *ps, const scalar dt) {
    // compute error
    const int nb_vertices = static_cast<int>(_owners.size());
    const scalar e = compute_energy(ps);
    const std::vector<Vector3> forces = compute_forces(ps, dt);
    scalar sum = 0;
    scalar total = 0;
    for (int i = 0; i < nb_vertices; ++i) {
        const Particle *p = ps->get(i);
        if (p->active) {
            total++;
            sum += glm::dot(forces[i], forces[i]);
            //Debug::SetColor(ColorBase::Red());
            //Debug::Line(p->position, p->position + forces[i]);
        }
    }
    sum /= total;
    DebugUI::Begin("Energy");
    DebugUI::Plot("energy", e, 200);
    DebugUI::Range("range", e);
    DebugUI::End();

    DebugUI::Begin("Forces");
    DebugUI::Value("Forces val", sum);
    DebugUI::Plot("Forces norm", sum, 200);
    DebugUI::Range("Forces range", sum);
    DebugUI::End();
}

scalar VBD_FEM::compute_energy(ParticleSystem *ps) const {
    scalar energy = 0;
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

            Matrix3x3 F = Jx * JX_inv[eid][i];
            energy += _material->get_energy(F) * V[eid][i];
        }
    }
    return energy;
}

std::vector<Vector3> VBD_FEM::compute_forces(ParticleSystem *ps, const scalar dt) const {
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

            Matrix3x3 F = Jx * JX_inv[eid][i];
            Matrix3x3 P = _material->get_pk1(F) * glm::transpose(JX_inv[eid][i]) * V[eid][i];
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

void VBD_FEM::solve_vertex(ParticleSystem *ps, const scalar dt, const int vid) {
    const int nb_owners = static_cast<int>(_owners[vid].size());
    Vector3 f_i = Unit3D::Zero();
    Matrix3x3 H_i = Matrix::Zero3x3();

    Particle *p = ps->get(vid);
    // sum all owner participation to force and hessian
    for (int i = 0; i < nb_owners; ++i) {
        const int owner = _owners[vid][i];
        const int ref_id = _ref_id[vid][i];
        solve_element(ps, owner, ref_id, f_i, H_i);
    }

    // Damping
    f_i -= _k_damp / dt * H_i * (p->position - p->last_position);
    H_i += _k_damp / dt * H_i;

    // Inertia
    f_i += -p->mass / (dt * dt) * (p->position - _y[vid]);
    H_i += Matrix3x3(p->mass / (dt * dt));

    const scalar detH = abs(glm::determinant(H_i));
    const Vector3 dx = detH > eps ? glm::inverse(H_i) * f_i : Unit3D::Zero();
    p->position += dx;
}

void VBD_FEM::solve_element(ParticleSystem *ps, const int eid, const int ref_id, Vector3 &f_i, Matrix3x3 &H_i) {
    const int nb_quadrature = _shape->nb_quadratures();
    const int nb_vert_elem = _shape->nb;
    std::vector<Matrix3x3> d2Psi_dF2(9);
    for (int i = 0; i < nb_quadrature; ++i) {
        Matrix3x3 Jx = Matrix::Zero3x3();
        for (int j = 0; j < nb_vert_elem; ++j) {
            const int vid = _topology[eid * nb_vert_elem + j];
            Jx += glm::outerProduct(ps->get(vid)->position, _shape->dN[i][j]);
        }

        Matrix3x3 F = Jx * JX_inv[eid][i];
        Vector3 dF_dx = glm::transpose(JX_inv[eid][i]) * _shape->dN[i][ref_id];

        // compute force
        Matrix3x3 P = _material->get_pk1(F);
        f_i -= P * dF_dx * V[eid][i];

        _material->get_sub_hessian(F, d2Psi_dF2);
        H_i += assemble_hessian(d2Psi_dF2, dF_dx) * V[eid][i];
    }
}

Matrix3x3 VBD_FEM::assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, const Vector3 dF_dx) {
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

void VBD_FEM::compute_inertia(ParticleSystem *ps, const scalar dt) {
    for (int i = 0; i < ps->nb_particles(); ++i) {
        Particle *p = ps->get(i);
        if (p->active) _y[i] = p->position + p->velocity * dt + (
                                   (p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt * dt;
    }
}

void VertexBlockDescent::chebyshev_acceleration(const int it, scalar &omega) {
    if (static_cast<int>(prev_prev_x.size()) == 0) {
        prev_prev_x.resize(nb_particles());
        prev_x.resize(nb_particles());
    }

    omega = compute_omega(omega, it);
    for (int i = 0; i < nb_particles(); ++i) {
        Particle *p = get(i);
        if (!p->active) continue;
        if (it >= 2) p->position = omega * (p->position - prev_prev_x[i]) + prev_prev_x[i];
        prev_prev_x[i] = prev_x[i];
        prev_x[i] = p->position;
    }
}

void VertexBlockDescent::step(const scalar dt) {
    const scalar sub_dt = dt / static_cast<scalar>(_sub_iteration);
    for (int i = 0; i < _sub_iteration; ++i) {
        _fem->compute_inertia(this, sub_dt);
        // get the first guess
        step_solver(sub_dt);
        scalar omega = 0;
        for (int j = 0; j < _iteration; ++j) {
            _fem->solve(this, sub_dt);
            chebyshev_acceleration(j, omega);
        }
        step_effects(sub_dt);
        step_constraint(sub_dt);
        update_velocity(sub_dt);
    }
    reset_external_forces();
}

void VertexBlockDescent::update_velocity(const scalar dt) const {
    for (Particle *p: this->_particles) {
        if (!p->active) continue;
        p->velocity = (p->position - p->last_position) / dt;
    }
}