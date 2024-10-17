#pragma once
#include <random>
#include <set>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"

struct Grid_Level {
    Grid_Level(FEM_Shape *shape,
             const std::vector<scalar> &masses,
             const std::vector<std::vector<Matrix3x3> > &jx_inv,
             const std::vector<std::vector<scalar> > &v, std::vector<int> ids) :
          _shape(shape), _masses(masses), _JX_inv(jx_inv), _V(v), _ids(ids) {}

    // const
    FEM_Shape *_shape; // element used in this grid level
    std::vector<int> _ids; // vertices that are used in this grid level
    std::vector<scalar> _masses; // for each vertice
    std::vector<std::vector<Matrix3x3> > _JX_inv; // per element
    std::vector<std::vector<scalar> > _V; // per element
};



struct P1_to_P2 {
    std::vector<int> ids;
    std::vector<std::pair<int,int>> edges;
    static std::pair<int,int> ref_edges[6];
    explicit P1_to_P2(const Mesh::Topology& topology) {

        for(int i = 0; i < topology.size(); i+=10) {
            for(int j = 0; j < 6; ++j) {
                ids.push_back(topology[i + 4 + j]);
                const int a = ref_edges[j].first, b = ref_edges[j].second;
                edges.emplace_back(topology[i + a], topology[i + b]);
            }
        }
    }

    virtual void prolongation(ParticleSystem* ps, std::vector<Vector3> dx) {
        for(int i = 0; i < ids.size(); i++) {
            const int a = edges[i].first, b = edges[i].second;
            ps->get(ids[i])->position += (dx[a] + dx[b]) * 0.5f;
        }

    }

    virtual ~P1_to_P2() = default;
};

std::pair<int,int> P1_to_P2::ref_edges[6] = {{0,1},{1,2},{2,0},{0,3},{1,3},{2,3}};

/// Faire la technique avec 1 maillage différent pour chaque niveau (par grave pour la mémoire)
/// Pour l'initialisation on considère que chaque niveau est une bonne approximation P1 == P2
/// Seul l'interpolation est au courant qu'il y a un lien entre les deux
/// Par grille : 1 topo, 1 système de particule, 1 forme
/// 1 VBD_FEM par niveau
/// 1 grid_interpolation entre chaque niveau
/// L'interpolation doit faire passer la déformation d'un niveau à un autre
/// Quand on décend de niveau, on change rien au niveau de la position.
/// Le guess y et x sont valides dans tous les niveaux
/// Donc on peut passer direct de P2 à P1 sans modificaiton du système de particule (il y a juste la masse qui va changer)
/// A chaque niveau on va modifier directement p mais on va aussi retenir la correction dx
/// à la fin de l'itération, on interpole dx vers le niveau au dessus
struct MG_VBD_FEM {
    MG_VBD_FEM(const Mesh::Topology &topology, const Mesh::Geometry &geometry, FEM_Shape *shape, FEM_ContinuousMaterial *material, scalar density) {
        // init global data and shape (P2)
        _shape = shape;
        _y = geometry;
        _material = material;

        // init neighboors for each particles (same for each level)
        build_neighboors(topology);
        // init constant for P2 => init grid[0]
        build_fem_const(topology, geometry, density, Tetra20);
        // init constant for P1 => init grid[1]
        build_fem_const(topology, geometry, density, Tetra);

        // init prolongation
        p1_to_p2 = new P1_to_P2(topology);
    }

    void build_fem_const(const Mesh::Topology &topology,  const Mesh::Geometry &geometry, scalar density, Element e) {
        const int nb_quadrature = _shape->nb_quadratures();
        const int nb_element = static_cast<int>(topology.size()) / _shape->nb;
        FEM_Shape* l_shape = get_fem_shape(e);
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
                    J += glm::outerProduct(geometry[topology[id + k]], _shape->dN[j][k]);
                }
                V[i][j] = abs(glm::determinant(J)) * _shape->weights[j];
                JX_inv[i][j] = glm::inverse(J);
                mass += V[i][j];
            }
            mass *= density / l_shape->nb;
            for (int k = 0; k < _shape->nb; ++k) {
                masses[topology[id + k]] = mass;
            }
        }
        std::vector<int> ids(s_ids.begin(), s_ids.end());
        grids.push_back(new Grid_Level(l_shape, masses, JX_inv, V, ids));
    }


    void build_neighboors(const Mesh::Topology &topology) {
        for (int i = 0; i < topology.size(); i += _shape->nb) {
            for (int j = 0; j < _shape->nb; ++j) {
                _owners[topology[i + j]].push_back(i / _shape->nb);
                _ref_id[topology[i + j]].push_back(j);
            }
        }
    }


    void solve(ParticleSystem *ps, scalar dt) {

        // coarse to refined
        for(int i = grids.size()-1; i >= 0; i++) {
            Grid_Level *grid = grids[i];
            std::fill(_dx.begin(), _dx.end(), Unit3D::Zero());
            for(int id : grid->_ids) {
                solve_vertex(ps, grid, dt, id);
            }
            // prolongatation
            p1_to_p2->prolongation(ps, _dx);
        }
    }

    void plot_residual(ParticleSystem *ps, scalar dt) {
        // can wait
    }

    scalar compute_energy(ParticleSystem *ps) const {
        // can wait
    }

    std::vector<Vector3> compute_forces(ParticleSystem *ps, scalar dt) const {
        // can wait
    }

    // int i = vertex index for ids
    void solve_vertex(ParticleSystem *ps, Grid_Level* grid, scalar dt, int i) {
        int vid = grid->_ids[i];
        // need current grid info
        const int nb_owners = static_cast<int>(_owners[vid].size());
        Vector3 f_i = Unit3D::Zero();
        Matrix3x3 H_i = Matrix::Zero3x3();
        Particle *p = ps->get(vid);
        for (int i = 0; i < nb_owners; ++i) {
            const int owner = _owners[vid][i];
            const int ref_id = _ref_id[vid][i];
            solve_element(ps, grid, owner, ref_id, f_i, H_i);
        }
        // Inertia
        f_i += -p->mass / (dt * dt) * (p->position - _y[vid]);
        H_i += Matrix3x3(p->mass / (dt * dt));
        const scalar detH = abs(glm::determinant(H_i));
        const Vector3 dx = detH > eps ? glm::inverse(H_i) * f_i : Unit3D::Zero();
        ps->get(vid)->position += dx;
        _dx[vid] += dx;
    }

    void solve_element(ParticleSystem *ps, Grid_Level* grid, int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i) {
        const int nb_quadrature = grid->_shape->nb_quadratures();
        const int nb_vert_elem =  grid->_shape->nb;
        const int nb_vert_elem_max =  _shape->nb;
        std::vector<Matrix3x3> d2Psi_dF2(9);
        for (int i = 0; i < nb_quadrature; ++i) {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = _topology[eid * nb_vert_elem_max + j];
                Vector3 p = ps->get(vid)->position; // gros problème ici on a besoin de faire le lien
                Jx += glm::outerProduct(p, _shape->dN[i][j]);
            }

            Matrix3x3 F = Jx * grid->_JX_inv[eid][i];
            Vector3 dF_dx = glm::transpose(grid->_JX_inv[eid][i]) * _shape->dN[i][ref_id];

            // compute force
            Matrix3x3 P = _material->get_pk1(F);
            f_i -= P * dF_dx * grid->_V[eid][i];

            _material->get_sub_hessian(F, d2Psi_dF2);
            H_i += assemble_hessian(d2Psi_dF2, dF_dx) *  grid->_V[eid][i];
        }
    }

    Matrix3x3 assemble_hessian(const std::vector<Matrix3x3> &d2W_dF2, Vector3 dF_dx);

    void compute_inertia(ParticleSystem *ps, scalar dt) {
        // normally we should make a better approximation but osef

        for (int i = 0; i < ps->nb_particles(); ++i) {
            const Particle *p = ps->get(i);
            _y[i] = p->position + p->velocity * dt + ((p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt * dt;
        }

    }

protected:
    std::vector<Grid_Level *> grids;
    P1_to_P2* p1_to_p2;
    FEM_ContinuousMaterial *_material;
    std::vector<Vector3> _y;
    std::vector<Vector3> _dx;
    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice
    FEM_Shape *_shape;
    Mesh::Topology _topology;
};
