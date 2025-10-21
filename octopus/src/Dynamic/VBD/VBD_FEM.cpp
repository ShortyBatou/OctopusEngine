#pragma once
#include <random>
#include <set>
#include <Dynamic/FEM/FEM_Generic.h>

#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include <Dynamic/VBD/VBD_FEM.h>
#include <Manager/Debug.h>
#include <Manager/Input.h>

void VBD_FEM::compute_inertia(VertexBlockDescent* vbd, const scalar dt) {
    for (int i = 0; i <  vbd->nb_particles(); ++i) {
        const Particle *p = vbd->get(i);
        _y[i] = p->position + p->velocity * dt + ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt * dt;
    }
}

VBD_FEM::VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, const Element e,
                 FEM_ContinuousMaterial *material, const scalar k_damp) : _k_damp(k_damp), _material(material) {
    _owners.resize(geometry.size());
    _ref_id.resize(geometry.size());
    data = build_data(topology, geometry, e);
    build_neighboors(topology);
    _y = geometry;
}

void VBD_FEM::build_neighboors(const Mesh::Topology &topology) {
    for (int i = 0; i < topology.size(); i += data->shape->nb) {
        for (int j = 0; j <  data->shape->nb; ++j) {
            _owners[topology[i + j]].push_back(i /  data->shape->nb);
            _ref_id[topology[i + j]].push_back(j);
        }
    }
}

VBD_FEM_Data* VBD_FEM::build_data(const Mesh::Topology &topology, const Mesh::Geometry &geometry, const Element e) {
    std::vector<std::vector<Matrix3x3>> JX_inv;
    std::vector<std::vector<scalar>> V;
    get_fem_const(e, geometry,topology, JX_inv, V);
    std::set<int> s_ids(topology.begin(), topology.end());
    const std::vector<int> ids(s_ids.begin(), s_ids.end());
    return new VBD_FEM_Data(get_fem_shape(e), topology, JX_inv, V, ids);
}

void VBD_FEM::solve(VertexBlockDescent *vbd, const scalar dt) {
    const int nb_vertices = static_cast<int>(_owners.size());
    std::vector<int> ids(nb_vertices);
    std::iota(ids.begin(), ids.end(), 0);
    //std::shuffle(ids.begin(), ids.end(), std::mt19937());
    for (int i = 0; i < nb_vertices; ++i) {
        Particle* p = vbd->get(i);
        if (p->active) solve_vertex(vbd, dt, ids[i], p->mass);
    }
    plot_residual(vbd, dt);
}

static int test_id = 0;

void VBD_FEM::solve_vertex(VertexBlockDescent *vbd, const scalar dt, const int vid, const scalar mass) {
    const int nb_owners = static_cast<int>(_owners[vid].size());
    Vector3 f_i = Unit3D::Zero();
    Matrix3x3 H_i = Matrix::Zero3x3();
    test_id = vid;
    Particle *p = vbd->get(vid);
    // sum all owner participation to force and hessian
    for (int i = 0; i < nb_owners; ++i) {
        const int owner = _owners[vid][i];
        const int ref_id = _ref_id[vid][i];
        solve_element(vbd, owner, ref_id, f_i, H_i);
    }

    // Damping
    f_i -= _k_damp / dt * H_i * (p->position - p->last_position);
    H_i += _k_damp / dt * H_i;

    // Inertia
    f_i -= mass / (dt * dt) * (p->position - _y[vid]);
    H_i += Matrix3x3(mass / (dt * dt));

    const scalar detH = abs(glm::determinant(H_i));
    const Vector3 dx = detH > eps ? glm::inverse(H_i) * f_i : Unit3D::Zero();
    p->position += dx;
}

void VBD_FEM::solve_element(VertexBlockDescent *ps, const int eid, const int ref_id, Vector3 &f_i, Matrix3x3 &H_i) {
    const int nb_quadrature = data->shape->nb_quadratures();
    const int nb_vert_elem = data->shape->nb;
    std::vector<Matrix3x3> d2Psi_dF2(9);
    for (int i = 0; i < nb_quadrature; ++i) {
        Matrix3x3 Jx = Matrix::Zero3x3();
        for (int j = 0; j < nb_vert_elem; ++j) {
            const int vid = data->topology[eid * nb_vert_elem + j];
            Jx += glm::outerProduct(ps->get(vid)->position, data->shape->dN[i][j]);
        }

        Matrix3x3 F = Jx * data->JX_inv[eid][i];
        Vector3 dF_dx = glm::transpose(data->JX_inv[eid][i]) * data->shape->dN[i][ref_id];

        // compute force
        Matrix3x3 P = _material->get_pk1(F);
        f_i -= P * dF_dx * data->V[eid][i];

        _material->get_sub_hessian(F, d2Psi_dF2);
        //if(test_id == 100 && i == 0) std::cout << d2Psi_dF2[1] << std::endl;

        H_i += assemble_hessian(d2Psi_dF2, dF_dx) * data->V[eid][i];
    }
}


void VBD_FEM::plot_residual(VertexBlockDescent *vbd, const scalar dt) const {
    // compute error
    const int nb_vertices = static_cast<int>(_owners.size());
    const std::vector<Vector3> forces = compute_forces(vbd, dt);
    scalar sum = 0;
    for (int i = 0; i < nb_vertices; ++i) {
        const Particle *p = vbd->get(i);
        if (p->active) {
            sum += glm::length(forces[i]);
        }
    }

    DebugUI::Begin("Forces");
    DebugUI::Value("Forces val", sum);
    DebugUI::Plot("Forces norm", sum, 200);
    DebugUI::Range("Forces range", sum);
    DebugUI::End();
}

std::vector<Vector3> VBD_FEM::compute_forces(VertexBlockDescent *vbd, const scalar dt) const {
    std::vector forces(vbd->nb_particles(), Unit3D::Zero());
    const int nb_vert_elem = data->shape->nb;
    const int nb_quadrature = data->shape->nb_quadratures();
    for (int e = 0; e < data->topology.size(); e += nb_vert_elem) {
        const int eid = e / nb_vert_elem;
        for (int i = 0; i < nb_quadrature; ++i) {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = data->topology[eid * nb_vert_elem + j];
                Jx += glm::outerProduct(vbd->get(vid)->position, data->shape->dN[i][j]);
            }

            Matrix3x3 F = Jx * data->JX_inv[eid][i];
            Matrix3x3 P = _material->get_pk1(F) * glm::transpose(data->JX_inv[eid][i]) * data->V[eid][i];
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = data->topology[eid * nb_vert_elem + j];
                forces[vid] -= P * data->shape->dN[i][j];
            }
        }
    }
    for (int i = 0; i < vbd->nb_particles(); ++i) {
        const Particle *p = vbd->get(i);
        forces[i] -= p->mass / (dt * dt) * (p->position - _y[i]);
    }
    return forces;
}

std::vector<scalar> VBD_FEM::compute_stress(VertexBlockDescent *vbd) const
{
    const int nb_element = static_cast<int>(data->JX_inv.size());
    std::vector<scalar> stress(nb_element,0);
    const int nb_quadrature = data->shape->nb_quadratures();
    const int nb_vert_elem = data->shape->nb;
    for(int eid = 0; eid < nb_element; ++eid)
    {
        for (int i = 0; i < nb_quadrature; ++i) {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = data->topology[eid * nb_vert_elem + j];
                Jx += glm::outerProduct(vbd->get(vid)->position, data->shape->dN[i][j]);
            }
            Matrix3x3 F = Jx * data->JX_inv[eid][i];
            // compute force
            Matrix3x3 P = _material->get_pk1(F);
            P = ContinuousMaterial::pk1_to_chauchy_stress(F, P);
            stress[eid] += ContinuousMaterial::von_mises_stress(P) * data->V[eid][i];
        }
    }
    return stress;
}

std::vector<scalar> VBD_FEM::compute_volume(VertexBlockDescent *vbd) const
{
    const int nb_element = static_cast<int>(data->JX_inv.size());
    std::vector<scalar> volume(nb_element,0);
    const int nb_quadrature = data->shape->nb_quadratures();
    const int nb_vert_elem = data->shape->nb;
    for(int eid = 0; eid < nb_element; ++eid)
    {
        for (int i = 0; i < nb_quadrature; ++i) {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = data->topology[eid * nb_vert_elem + j];
                Jx += glm::outerProduct(vbd->get(vid)->position, data->shape->dN[i][j]);
            }
            volume[eid] += abs(glm::determinant(Jx)) * data->shape->weights[i];
        }
    }
    return volume;
}

std::vector<scalar> VBD_FEM::compute_colume_diff(VertexBlockDescent *vbd) const
{
    std::vector<scalar> volume_diff = compute_volume(vbd);
    const int nb_element = static_cast<int>(data->JX_inv.size());
    const int nb_quadrature = data->shape->nb_quadratures();
    for(int eid = 0; eid < nb_element; ++eid)
    {
        for (int i = 0; i < nb_quadrature; ++i) {
            volume_diff[eid] -= data->V[eid][i];
        }
    }
    return volume_diff;
}

std::vector<scalar> VBD_FEM::compute_inverted(VertexBlockDescent *vbd) const
{
    const int nb_element = static_cast<int>(data->JX_inv.size());
    std::vector<scalar> inverted(nb_element,0);
    const int nb_quadrature = data->shape->nb_quadratures();
    const int nb_vert_elem = data->shape->nb;
    for(int eid = 0; eid < nb_element; ++eid)
    {
        for (int i = 0; i < nb_quadrature; ++i) {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = data->topology[eid * nb_vert_elem + j];
                Jx += glm::outerProduct(vbd->get(vid)->position, data->shape->dN[i][j]);
            }

            if(glm::determinant(Jx * data->JX_inv[eid][i]) < 0) inverted[eid] = 0;;
        }
    }
    return inverted;
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
