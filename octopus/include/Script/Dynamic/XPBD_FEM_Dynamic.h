#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"
#include "Dynamic/PBD/PositionBasedDynamic.h"
#include "Dynamic/PBD/XPBD_FEM_Generic.h"
#include "Script/Dynamic/FEM_Dynamic.h"

class XPBD_FEM_Dynamic : public FEM_Dynamic {
public:
    XPBD_FEM_Dynamic(scalar density, 
        scalar young, scalar poisson, 
        Material material, 
        int iteration = 1, int sub_iteration = 30, 
        scalar global_damping = 0,
        PBDSolverType type = GaussSeidel) : 
        FEM_Dynamic(density, young, poisson, material, sub_iteration), 
        _iteration(iteration),
        _type(type),
        _global_damping(global_damping)
    { }

    virtual ~XPBD_FEM_Dynamic() { }

    int get_iteration() { return _iteration; }
    int get_sub_iteration() { return _sub_iteration; }
    void set_iterations(int it, int sub_it) {
        _iteration = it;
        _sub_iteration = sub_it;
        PBD_System* pbd = static_cast<PBD_System*>(this->_ps);
        pbd->_nb_step = it;
        pbd->_nb_substep = sub_it;
    }

protected:
    virtual ParticleSystem* build_particle_system() override {
        return new PBD_System(new EulerSemiExplicit(1.f), _iteration, _sub_iteration, _type, _global_damping);
    }

    virtual std::vector<FEM_Generic*> build_element(const std::vector<int>& ids, Element type, scalar& volume) override {
        std::vector<PBD_ContinuousMaterial*> materials = get_pbd_materials(_material, _young, _poisson);
        auto pbd = static_cast<PBD_System*>(_ps);
        XPBD_FEM_Generic* fem;
        std::vector<FEM_Generic*> fems;
        for (PBD_ContinuousMaterial* m : materials) {
            fem = new XPBD_FEM_Generic(ids, m, get_fem_shape(type));
            fems.push_back(fem);
            pbd->add_xpbd_constraint(fem);
        }
        volume = fem->compute_volume(fem->get_particles(_ps->particles()));
        return fems;
    }
    
public:
    scalar _global_damping;
    int _iteration;
    PBDSolverType _type;
};