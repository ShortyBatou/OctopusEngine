#pragma once
#include "Dynamic/Base/ParticleSystem.h"
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/Base/Solver.h"
#include <algorithm>
#include <random>
#include <iterator>
enum PBDSolverType {
    GaussSeidel, Jacobi, GaussSeidel_RNG
};

struct PBD_Thread_Graph {
    void create_graph_color(std::vector<Particle*>& parts, std::vector<std::vector<XPBD_Constraint*>>& g_constraints) {
        unsigned int max_color = 0;
        std::vector<unsigned int> c_colors(g_constraints.size());
        std::vector<std::set<unsigned int>> parts_colors(parts.size());
        for (unsigned int i = 0; i < g_constraints.size(); ++i) {
            if (g_constraints[i].size() == 0) continue;
            Mesh::Topology c_topo = g_constraints[i][0]->ids();
            unsigned int current_color = 0;
            
            /// get neighbors' color
            std::set<unsigned int> unavailable_colors;
            for (unsigned int id : c_topo) 
            for (unsigned int p_color : parts_colors[id]) {
                unavailable_colors.insert(p_color);
            }
            
            // find the minimum available color
            for (unsigned int u_color : unavailable_colors) {
                if (u_color == current_color) current_color++;
                else break;
            }
            
            max_color = std::max(current_color, max_color);

            // update vertices color
            for (unsigned int id : c_topo) {
                parts_colors[id].insert(current_color);
            }

            c_colors[i] = current_color;
        }
        std::cout << "NB COLORS = " << max_color << std::endl;
        // create tread group

    }
};


struct PBD_System : public ParticleSystem {
    PBD_System(Solver* solver, unsigned int nb_step, unsigned int nb_substep = 1, PBDSolverType solver_type = GaussSeidel, scalar global_damping = scalar(0)) :
        ParticleSystem(solver), _nb_step(nb_step), _nb_substep(nb_substep), _type(solver_type), _global_damping(global_damping)
    { 
        _groups.push_back(std::vector<XPBD_Constraint*>());
    }

    virtual void step(const scalar dt) override {
        //if (!_init) {
        //    PBD_Thread_Graph graph;
        //    graph.create_graph_color(this->particles(), this->_groups);
        //    _init = true;
        //}
        
        scalar h = dt / (scalar)_nb_substep;
        for (unsigned int i = 0; i < _nb_substep; i++)
        {
            this->step_solver(h);
            this->reset_lambda();
            for (unsigned int j = 0; j < _nb_step; ++j) {
                if (_type == Jacobi)
                    step_constraint_jacobi(h);
                if(_type == GaussSeidel_RNG) 
                    step_constraint_gauss_seidel_rng(h);
                else
                    step_constraint_gauss_seidel(h);
            }

            this->step_constraint(dt); // optional 

            this->step_effects(dt); // optional

            this->update_velocity(h);
        }
        this->reset_external_forces();
    }

    virtual ~PBD_System() {
        clear_xpbd_constraints();
    }

    void clear_xpbd_constraints() {
        for (XPBD_Constraint* c : _xpbd_constraints) delete c;
        _xpbd_constraints.clear();
    }

    unsigned int add_xpbd_constraint(XPBD_Constraint* constraint) {
        _xpbd_constraints.push_back(constraint);
        _xpbd_constraints.back()->init(this->_particles);
        _groups[_groups.size() - 1].push_back(constraint);
        return _xpbd_constraints.size();
    }

    void new_group() {
        _groups.push_back(std::vector<XPBD_Constraint*>());
    }

    void draw_debug_xpbd() {
        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            xpbd->draw_debug(this->_particles);
        }
    }

public:
    virtual void reset_lambda() {
        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            xpbd->set_lambda(0);
        }
    }

    virtual void update_velocity(scalar dt) {
        for(Particle* p : this->_particles)
        {
            if (!p->active) continue;
            p->velocity = (p->position - p->last_position) / dt;
            
            scalar norm_v = glm::length(p->velocity);
            if (norm_v > 1e-12) {
                scalar damping = -norm_v * std::min(scalar(1), _global_damping * dt * p->inv_mass);
                p->velocity += glm::normalize(p->velocity) * damping ;
            }
        }
    }

    virtual void step_constraint_gauss_seidel(const scalar dt) {

        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            if (!xpbd->active()) continue;

            // compute corrections
            xpbd->apply(this->_particles, dt);

            // apply correction dt_p on particles' position
            for (unsigned int id : xpbd->ids()) {
                this->_particles[id]->position += this->_particles[id]->force; // here: force = delta x
                this->_particles[id]->force *= 0;
            }
        }
    }

    virtual void step_constraint_gauss_seidel_rng(const scalar dt) {
        
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(_groups), std::end(_groups), rng);

        for (std::vector<XPBD_Constraint*>& group : _groups) {
            std::reverse(group.begin(), group.end());
            for (XPBD_Constraint* xpbd : group) {
                if (!xpbd->active()) continue;
                
                // compute corrections
                xpbd->apply(this->_particles, dt);

                // apply correction dt_p on particles' position
                for (unsigned int id : xpbd->ids()) {
                    this->_particles[id]->position += this->_particles[id]->force;
                    this->_particles[id]->force *= 0;
                }
            }
        }
    }

    virtual void step_constraint_jacobi(const scalar dt) {
        std::vector<unsigned int> counts(this->_particles.size(), 0);

        for (XPBD_Constraint* xpbd : _xpbd_constraints) {
            if (!xpbd->active()) continue;
            xpbd->apply(this->_particles, dt); // if xpbd

            for (unsigned int id : xpbd->ids()) {
                counts[id]++;
            }
        }

        for (unsigned int i = 0; i < this->_particles.size(); ++i) 
        {
            Particle*& part = this->_particles[i];
            part->position += part->force / scalar(counts[i]);
            part->force *= 0;
        }
    }

public:
    bool _init = false;
    scalar _global_damping;
    unsigned int _nb_step, _nb_substep;
    PBDSolverType _type;
protected:
    std::vector<std::vector<XPBD_Constraint*>> _groups;
    std::vector<XPBD_Constraint*> _xpbd_constraints;
};
