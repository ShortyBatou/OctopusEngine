#pragma once
#include <random>
#include <set>
#include <Dynamic/FEM/FEM_Shape.h>
#include "Core/Base.h"
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"

struct Grid_Level {
    Grid_Level(FEM_Shape *shape, const Mesh::Geometry &positions, const Mesh::Topology &topology,
               const std::vector<scalar> &masses,
               const std::vector<std::vector<Matrix3x3> > &jx_inv, const std::vector<std::vector<scalar> > &v,
               const Mesh::Geometry &y)
        : _shape(shape),
          _masses(masses),
          _JX_inv(jx_inv),
          _V(v),
          _y(y), _dx(Mesh::Geometry(y.size(), Unit3D::Zero())) {
    }

    // const
    FEM_Shape *_shape; // element used in this grid level
    std::vector<int> ids; // vertices that are used in this grid level
    std::vector<scalar> _masses; // for each vertice
    std::vector<std::vector<Matrix3x3> > _JX_inv; // per element
    std::vector<std::vector<scalar> > _V; // per element

    // dynamic
    Mesh::Geometry _y; // each integration
    Mesh::Geometry _dx;
};

struct Grid_Interpolation {
    virtual ~Grid_Interpolation() = default;

    virtual void restriction(const Grid_Level *gH, Grid_Level *g2H) = 0;

    virtual void prolongation(const Grid_Level *g2H, Grid_Level *gH) = 0;
};

/// Faire la technique avec 1 maillage différent pour chaque niveau (par grave pour la mémoire)
/// Pour l'initialisation on considère que chaque niveau est une bonne approximation P1 == P2
/// Seul l'interpolation est au courant qu'il y a un lien entre les deux
/// Par grille : 1 topo, 1 système de particule, 1 forme
/// 1 VBD_FEM par niveau
/// 1 grid_interpolation entre chaque niveau
/// L'interpolation doit faire passer la déformation d'un niveau à un autre
/// Prec_position => position
/// Au début de l'itération, il faut faire passer l'information de déformation à tous les niveau d'en dessous
/// => comment faire ? on regarde la différence entre la position initiale de notre maillage le plus détaillé et on le fait passer de niveau en niveau
/// => autrement dit on va utiliser u(X) pour partager l'information ! new_position = init_position + sum u(x_i) w_i
struct MG_VBD_FEM {
    MG_VBD_FEM(const Mesh::Topology &topology, ParticleSystem *ps, FEM_Shape *shape, FEM_ContinuousMaterial *material) {
        // init global shape (P2)
        // init neighboors for each particles (same for each level)
        // init constant for P2 => init grid[0]
        // init constant for P1 => init grid[1]
    }

    void build_fem_const(const Mesh::Topology &topology, ParticleSystem *ps, FEM_Shape *shape) {
        // in : topology, density, shape type
        // resume : init grid
        // we suppose that P2 == P1 at rest state

        // get shape
        // build V  from shape and topo
        // build JX from shape and topo
        // build M  from shape and topo
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
        // Type : Cascade
        // for each grid start at coarse
        //      Set grid->dx to 0
        //      for each id in grid
        //          solve n time for grid_n to get dx
        //      prolongation of dx to grid_n-1
        // apply grid[0]->_dx on ps
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

    void solve_vertex(ParticleSystem *ps, Grid_Level* grid, scalar dt, int id) {
        int vid = grid->ids[id];
        // need current grid info
        const int nb_owners = static_cast<int>(_owners[vid].size());
        Vector3 f_i = Unit3D::Zero();
        Matrix3x3 H_i = Matrix::Zero3x3();
        Particle *p = ps->get(vid);
        for (int i = 0; i < nb_owners; ++i) {
            const int owner = _owners[vid][i];
            const int ref_id = _ref_id[vid][i];
            solve_element(ps, grid, id, owner,ref_id, f_i, H_i);
        }
        // Inertia
        f_i += -p->mass / (dt * dt) * (p->position - grid->_y[id]);
        H_i += Matrix3x3(p->mass / (dt * dt));
        const scalar detH = abs(glm::determinant(H_i));
        const Vector3 dx = detH > eps ? glm::inverse(H_i) * f_i : Unit3D::Zero();
        grid->_dx[id] += dx;
    }

    void solve_element(ParticleSystem *ps, Grid_Level* grid, int id,  int eid, int ref_id, Vector3 &f_i, Matrix3x3 &H_i) {
        const int nb_quadrature = grid->_shape->nb_quadratures();
        const int nb_vert_elem =  grid->_shape->nb;
        const int nb_vert_elem_max =  _shape->nb;
        std::vector<Matrix3x3> d2Psi_dF2(9);
        for (int i = 0; i < nb_quadrature; ++i) {
            Matrix3x3 Jx = Matrix::Zero3x3();
            for (int j = 0; j < nb_vert_elem; ++j) {
                const int vid = _topology[eid * nb_vert_elem_max + j];
                Vector3 p = ps->get(vid)->position + grid->_dx[id]; // gros problème ici on a besoin de faire le lien
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
        for (Grid_Level *grid: grids) {
            for (int i = 0; i < grid->ids.size(); ++i) {
                const int id = grid->ids[i];
                const Particle *p = ps->get(id);
                grid->_y[i] = p->position + p->velocity * dt + (
                                  (p->force + p->external_forces) * p->inv_mass + Dynamic::gravity()) * dt * dt;
            }
        }
    }

protected:
    std::vector<Grid_Level *> grids;

    FEM_ContinuousMaterial *_material;

    std::vector<std::vector<int> > _owners; // for each vertice
    std::vector<std::vector<int> > _ref_id; // for each vertice
    FEM_Shape *_shape;
    Mesh::Topology _topology;
};
