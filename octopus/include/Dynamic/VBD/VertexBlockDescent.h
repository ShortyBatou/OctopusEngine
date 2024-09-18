#pragma once
#include <Dynamic/FEM/FEM_Shape.h>

#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
struct VBD_FEM
{
    VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape* shape, FEM_ContinuousMaterial* material) :
        _shape(shape), _material(material), _topology(topology)
    {
        build_fem_const(topology, geometry);
        build_neighboors(topology);
    }

    void build_neighboors(const Mesh::Topology &topology)
    {
        for (int i = 0; i < topology.size(); i += _shape->nb) {
            for (int j = 0; j < _shape->nb; ++j) {
                _owners[topology[i + j]].push_back(i / _shape->nb);
                _ref_id[topology[i+j]].push_back(j);
            }
        }
    }

    void build_fem_const(const Mesh::Topology &topology, const Mesh::Geometry &geometry)
    {
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

    void solve(ParticleSystem* ps, const scalar dt)
    {
        const int nb_vertices = static_cast<int>(_owners.size());
        for(int i = 0; i < nb_vertices; ++i)
        {
            solve_vertex(ps, dt, i);
        }
    }

    void solve_vertex(ParticleSystem* ps, const scalar dt, const int vid)
    {
        const int nb_owners = static_cast<int>(_owners[vid].size());
        Vector3 f_i = Unit3D::Zero();
        Matrix3x3 H_i = Matrix::Zero3x3();

        Particle* p = ps->get(vid);
        // sum all owner participation to force and hessian
        for(int i = 0; i < nb_owners; ++i)
        {
            const int owner = _owners[vid][i];
            const int ref_id = _ref_id[vid][i];
            solve_element(ps, owner, ref_id, f_i, H_i);
        }

        //f_i += -p->mass / (dt * dt) * (p->position - p->y);
        //H_i += Matrix3x3(p->mass / (dt * dt));
        const scalar detH = glm::determinant(H_i);
        const Vector3 dx = detH > eps ? glm::inverse(H_i) * f_i : Unit3D::Zero();
        p->position += dx;
    }

    void solve_element(const ParticleSystem* ps, const int eid, const int ref_id, Vector3& f_i, Matrix3x3& H_i)
    {
        const int nb_quadrature = _shape->nb_quadratures();
        const int nb_vert_elem = _shape->nb;
        std::vector<Matrix3x3> d2Psi_dF2(9);
        for(int i = 0; i < nb_quadrature; ++i)
        {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for(int j = 0; j < nb_vert_elem; ++j)
            {
                const int vid = _topology[eid * nb_vert_elem + i];
                Jx += glm::outerProduct(ps->get(vid)->position, _shape->dN[i][j]);
            }

            Matrix3x3 F = Jx * JX_inv[eid][i];
            Vector3 dF_dx = JX_inv[eid][i] * _shape->dN[i][ref_id];

            // compute force
            Matrix3x3 P = _material->get_pk1(F);
            f_i += P * dF_dx * V[eid][i];

            _material->get_sub_hessian(F, d2Psi_dF2);
            H_i += assemble_hessian(d2Psi_dF2, dF_dx) * V[eid][i];
        }
    }

    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3>& d2W_dF2, const Vector3 dF_dx)
    {
        Matrix3x3 H = Matrix::Zero3x3();
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                Matrix3x3 H_kl;
                for(int l = 0; l < 3; ++l) {
                    for(int k = 0; k < 3; ++k) {
                        H_kl[k][l] = d2W_dF2[k+l*3][i][j];
                    }
                }
                H[i][j] = glm::dot(dF_dx, H_kl * dF_dx);
            }
        }
        return H;
    }
protected:
    Mesh::Topology _topology;
    std::vector<std::vector<int>> _owners; // for each vertice
    std::vector<std::vector<int>> _ref_id; // for each vertice

    FEM_ContinuousMaterial* _material;
    FEM_Shape* _shape;
    std::vector<std::vector<Matrix3x3>> JX_inv; // per element
    std::vector<std::vector<scalar>> V; // per element
};

struct VertexBlockDescent final : ParticleSystem
{
    explicit VertexBlockDescent(Solver *solver, const int iteration, const int sub_iteration, const scalar rho)
        : ParticleSystem(solver), _iteration(iteration), _sub_iteration(sub_iteration), _rho(rho) {
    }

    void step(const scalar dt) override
    {
        const scalar sub_dt = dt / static_cast<scalar>(_sub_iteration);

        for(int i = 0; i < _sub_iteration; ++i)
        {
            step_solver(sub_dt);
            for(int j = 0; j < _iteration; ++j)
            {
                step_vbd(sub_dt);
            }
            step_effects(sub_dt);
            step_constraint(sub_dt);
            reset_external_forces();
        }
    }

    void step_vbd(const scalar dt)
    {

    }

protected:
    std::vector<Vector3> _y;
    VBD_FEM* _fem;
    int _iteration;
    int _sub_iteration;
    scalar _rho;
};
