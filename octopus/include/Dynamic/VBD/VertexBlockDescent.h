#pragma once
#include <random>
#include <Dynamic/FEM/FEM_Shape.h>
#include <Manager/Debug.h>
#include <Manager/Input.h>

#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
struct VBD_FEM
{
    VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape* shape, FEM_ContinuousMaterial* material) :
        _shape(shape), _material(material), _topology(topology)
    {
        _owners.resize(geometry.size());
        _ref_id.resize(geometry.size());
        build_fem_const(topology, geometry);
        build_neighboors(topology);
        _y = geometry;
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
        std::vector<int> ids(nb_vertices);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), std::mt19937());
        for(int i = 0; i < nb_vertices; ++i)
        {
            solve_vertex(ps, dt, ids[i]);
        }
        const scalar e = compute_energy(ps);
        const std::vector<Vector3> forces = compute_forces(ps);
        scalar sum = 0;
        for(int i = 0; i < nb_vertices; ++i)
        {
            sum += glm::length2(forces[i]);
        }
        sum /= static_cast<scalar>(nb_vertices);

        DebugUI::Begin("Energy");
        DebugUI::Plot("energy", e, 200);
        DebugUI::Range("range", e);
        DebugUI::Plot("Forces norm", sum, 200);
        DebugUI::Range("Forces range", sum);
        DebugUI::End();
    }

    scalar compute_energy(ParticleSystem* ps) const
    {
        scalar energy = 0;
        const int& nb_vert_elem = _shape->nb;
        const int nb_quadrature = _shape->nb_quadratures();
        for(int e = 0; e < _topology.size(); e+= _shape->nb)
        {
            const int eid = e / _shape->nb;
            for(int i = 0; i < nb_quadrature; ++i)
            {
                Matrix3x3 Jx = Matrix::Zero3x3();
                for(int j = 0; j < nb_vert_elem; ++j)
                {
                    const int vid = _topology[eid * nb_vert_elem + j];
                    Jx += glm::outerProduct(ps->get(vid)->position, _shape->dN[i][j]);
                }

                Matrix3x3 F = Jx * JX_inv[eid][i];
                energy += _material->get_energy(F) * V[eid][i];
            }
        }
        return energy;
    }

    std::vector<Vector3> compute_forces(ParticleSystem* ps) const
    {
        std::vector<Vector3> forces(ps->nb_particles(), Unit3D::Zero());
        const int& nb_vert_elem = _shape->nb;
        const int nb_quadrature = _shape->nb_quadratures();
        for(int e = 0; e < _topology.size(); e+= _shape->nb)
        {
            const int eid = e / _shape->nb;
            for(int i = 0; i < nb_quadrature; ++i)
            {
                Matrix3x3 Jx = Matrix::Zero3x3();
                for(int j = 0; j < nb_vert_elem; ++j)
                {
                    const int vid = _topology[eid * nb_vert_elem + j];
                    Jx += glm::outerProduct(ps->get(vid)->position, _shape->dN[i][j]);
                }

                Matrix3x3 F = Jx * JX_inv[eid][i];
                Matrix3x3 P = _material->get_pk1(F) * glm::transpose(JX_inv[eid][i]) * V[eid][i];
                for(int j = 0; j < nb_vert_elem; ++j)
                {
                    const int vid = _topology[eid * nb_vert_elem + j];
                    forces[vid] += P * _shape->dN[i][j];
                }
            }
        }
        return forces;
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

        f_i += -p->mass / (dt * dt) * (p->position - _y[vid]);
        H_i += Matrix3x3(p->mass / (dt * dt));
        const scalar detH = abs(glm::determinant(H_i));
        const Vector3 dx = detH > eps ? glm::inverse(H_i) * f_i : Unit3D::Zero();
        p->position += dx;
        //Debug::Line(p->position, p->position + glm::normalize(p->force) * 0.1f);
    }

    void solve_element(ParticleSystem* ps, const int eid, const int ref_id, Vector3& f_i, Matrix3x3& H_i)
    {
        const int nb_quadrature = _shape->nb_quadratures();
        const int nb_vert_elem = _shape->nb;
        std::vector<Matrix3x3> d2Psi_dF2(9);
        for(int i = 0; i < nb_quadrature; ++i)
        {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for(int j = 0; j < nb_vert_elem; ++j)
            {
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
    void compute_inertia(ParticleSystem* ps, const scalar dt)
    {
        for(int i = 0; i < ps->nb_particles(); ++i)
        {
            Particle* p = ps->get(i);
            _y[i] = p->position + p->velocity * dt + ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt * dt;
        }
    }
protected:
    std::vector<Vector3> _y; // inertia
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
        : ParticleSystem(solver), _fem(nullptr), _iteration(iteration), _sub_iteration(sub_iteration), _rho(rho) {
    }

    [[nodiscard]] scalar compute_omega(const scalar omega, const int it) const
    {
        if(it == 0) return 1.f;
        if(it == 1) return 2.f / (2.f - _rho * _rho);
        return 4.f / (4.f - _rho * _rho * omega);
    }

    void chebyshev_acceleration(const int it, scalar& omega)
    {
        if(static_cast<int>(prev_prev_x.size()) == 0)
        {
            prev_prev_x.resize(nb_particles());
            prev_x.resize(nb_particles());
        }

        omega = compute_omega(omega, it);
        for(int i = 0; i < nb_particles(); ++i)
        {
            Particle* p = get(i);
            if(!p->active) continue;
            if(it >= 2) p->position = omega * (p->position - prev_prev_x[i]) + prev_prev_x[i];
            prev_prev_x[i] = prev_x[i];
            prev_x[i] = p->position;
        }
    }

    void step(const scalar dt) override
    {
        if(Input::Down(Key::W)) _iteration++;
        if(Input::Down(Key::S)) _iteration--;

        if(Input::Down(Key::Q)) _sub_iteration++;
        if(Input::Down(Key::A)) _sub_iteration--;

        if(Input::Down(Key::W) || Input::Down(Key::S) || Input::Down(Key::A) || Input::Down(Key::Q))
        {
            std::cout << "Iteration: " << _iteration << " SubIteration:" << _sub_iteration << std::endl;
        }

        const scalar sub_dt = dt / static_cast<scalar>(_sub_iteration);
        for(int i = 0; i < _sub_iteration; ++i)
        {
            _fem->compute_inertia(this, sub_dt);
            // get the first guess
            step_solver(sub_dt);
            scalar omega = 0;
            for(int j = 0; j < _iteration; ++j)
            {
                _fem->solve(this, sub_dt);
                //chebyshev_acceleration(j, omega);
                step_constraint(sub_dt);
            }
            step_effects(sub_dt);
            step_constraint(sub_dt);
            update_velocity(sub_dt);
        }
        reset_external_forces();
    }

    void update_velocity(const scalar dt) const {
        for (Particle *p: this->_particles) {
            if (!p->active) continue;
            p->velocity = (p->position - p->last_position) / dt;
            const scalar norm_v = glm::length(p->velocity);
            if (norm_v > 1e-12) {
                const scalar damping = -norm_v * std::min(1.f, 100.f * dt * p->inv_mass);
                p->velocity += glm::normalize(p->velocity) * damping;
            }
        }
    }

    void setFEM(VBD_FEM* fem)
    {
        _fem = fem;
    }

    ~VertexBlockDescent() override = default;

protected:
    VBD_FEM* _fem;
    std::vector<Vector3> prev_x;
    std::vector<Vector3> prev_prev_x;

    int _iteration;
    int _sub_iteration;
    scalar _rho;
};
