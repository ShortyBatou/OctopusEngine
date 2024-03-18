#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"
#include "Dynamic/PBD/PositionBasedDynamic.h"
#include "Dynamic/PBD/XPBD_FEM_Generic.h"
#include "Script/Dynamic/ParticleSystemDynamic.h"
#include "Dynamic/PBD/XPBD_FEM_Tetra.h"
#include "Dynamic/PBD/XPBD_FEM_SVD_Generic.h"

class XPBD_FEM_Dynamic : public ParticleSystemDynamic {
public:
    XPBD_FEM_Dynamic(scalar density, 
        scalar young, scalar poisson, 
        Material material, 
        unsigned int iteration = 1, unsigned int sub_iteration = 30, 
        scalar global_damping = 0,
        PBDSolverType type = GaussSeidel) : 
        ParticleSystemDynamic(density),  _young(young), _poisson(poisson), 
        _material(material), 
        _iteration(iteration), _sub_iteration(sub_iteration), 
        _type(type), 
        _density(density), 
        _global_damping(global_damping)
    { }

    virtual void late_init() override {
        this->update_mesh();
    }

    virtual void update() {
        Time::Tic();
        this->_ps->step(Time::Fixed_DeltaTime());
        //scalar residual = _pbd->get_residual(Time::Fixed_DeltaTime());
        //if (residual > 10e-5 && _sub_iteration < 50) {
        //    set_iterations(_iteration, _sub_iteration + 1);
        //}

        //DebugUI::Begin("[" + std::to_string(this->_entity->id()) + "] XPBD FEM Dual Residual");
        //{
        //    DebugUI::Value("Residual", residual);
        //    DebugUI::Range("Range", residual);
        //    DebugUI::Plot("Plot", residual, 60);
        //}
        //DebugUI::End();

        this->_ps->draw_debug_constraints();
        this->update_mesh();
    }

    virtual ~XPBD_FEM_Dynamic() {
    }

    unsigned int get_iteration() { return _iteration; }
    unsigned int get_sub_iteration() { return _sub_iteration; }
    void set_iterations(unsigned int it, unsigned int sub_it) {
        _iteration = it;
        _sub_iteration = sub_it;
        PBD_System* pbd = static_cast<PBD_System*>(this->_ps);
        pbd->_nb_step = it;
        pbd->_nb_substep = sub_it;
    }

protected:
    virtual ParticleSystem* build_particle_system() override {
        return new PBD_System(new EulerSemiExplicit(Vector3(0.,-9.81,0.) * 1.f, 1.f), _iteration, _sub_iteration, _type, _global_damping);
    }

    virtual void build_dynamic() {
        _pbd = static_cast<PBD_System*>(this->_ps);
        for (Particle* p : _pbd->particles()) p->mass = 0;

        scalar t_volume = 0;
        unsigned int nb_element = 0;

        for (auto& topo : _mesh->topologies()) {
            Element type = topo.first;
            unsigned int nb = elem_nb_vertices(type);
            std::vector<unsigned int> ids(nb);
            nb_element += topo.second.size() / nb;
            for (unsigned int i = 0; i < topo.second.size(); i += nb) {
                for (unsigned int j = 0; j < nb; ++j) {
                    ids[j] = topo.second[i + j];
                }

                scalar volume = 0;

                //std::vector<PBD_ContinuousMaterial*> materials;
                //materials = get_pbd_materials(_material, _young, _poisson);
                //for (PBD_ContinuousMaterial* m : materials) {                    
                //    XPBD_FEM_Generic* fem = new XPBD_FEM_Generic(ids.data(), m, get_fem_shape(type));
                //    fems.push_back(fem);
                //    _pbd->add_xpbd_constraint(fem);
                //    volume += fem->get_init_volume();
                //}
                //
                //volume /= materials.size();
                
                SVD_ContinuousMaterial* m = get_svd_materials(_material, _young, _poisson);
                XPBD_FEM_SVD_Generic* fem = new XPBD_FEM_SVD_Generic(ids.data(), m, get_fem_shape(type));
                _pbd->add_xpbd_constraint(fem);
                volume += fem->init_volume;

                t_volume += volume;
                for (unsigned int j = 0; j < nb; ++j) {
                    Particle* p = _pbd->get(ids[j]);
                    p->mass += _density * volume / nb;
                }

                // create a constriant group (order garanty) 
                if (i + nb < topo.second.size()) _pbd->new_group();
            }
        }

        scalar total_mass = 0;
        for (Particle* p : _pbd->particles()) {
            p->inv_mass = scalar(1) / p->mass;
            total_mass += p->mass;
        }

        std::cout << "XPBD FEM BUILDED : VERTICES = " << _pbd->particles().size() << "   ELEMENTS = " << nb_element << "   MASS = " << total_mass << "   VOLUME = " << t_volume <<  std::endl;
    }
    
public:
    PBD_System* _pbd;
    std::vector<XPBD_FEM_Generic*> fems;

    scalar _global_damping;
    scalar _density;
    scalar _young, _poisson;
    unsigned int _iteration, _sub_iteration;
    Material _material;
    PBDSolverType _type;
};