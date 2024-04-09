#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"

#include "Manager/Input.h"
#include "Dynamic/FEM/FEM.h"
#include "Dynamic/Base/Constraint.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Generic.h"

class FEM_Dynamic : public ParticleSystemDynamic {
public:
    FEM_Dynamic(scalar density, scalar young, scalar poisson, Material material, int sub_iteration = 30)
        : ParticleSystemDynamic(density), _young(young), _poisson(poisson), _material(material), _sub_iteration(sub_iteration), _density(density) {
    }

    virtual void update() override {
        Time::Tic();
        this->_ps->step(Time::Fixed_DeltaTime());
        scalar tac = Time::Tac() * 1000.f;

        scalar init_total_volume = 0;
        scalar total_volume = 0;
        for (FEM_Generic* fem : fems) {
            init_total_volume += fem->init_volume;
            total_volume += fem->volume;
        }
        std::cout << "FEM : " << tac << " ms" << "  " << init_total_volume << " :: " << total_volume <<  std::endl;
        
        for (int i = 0; i < _mesh->nb_vertices(); ++i) {
            _mesh->geometry()[i] = this->_ps->get(i)->position;
        }
    }


    virtual ParticleSystem* build_particle_system() override {
        return new FEM_System(new EulerSemiExplicit(Vector3(0., -9.81, 0.)*1.f, 0.9995f), _sub_iteration);
    }


    virtual void build_dynamic() {
        FEM_System* s_fem= static_cast<FEM_System*>(this->_ps);
        for (Particle* p : s_fem->particles()) p->mass = 0;

        for (auto& topo : _mesh->topologies()) {
            Element type = topo.first;
            int nb = elem_nb_vertices(type);
            std::vector<int> ids(nb);
            for (int i = 0; i < topo.second.size(); i += nb) {
                for (int j = 0; j < nb; ++j) {
                    ids[j] = topo.second[i + j];
                }
                //FEM_Generic* fem = new FEM_Generic(ids.data(), get_fem_material(_material, _young, _poisson), get_fem_shape(type));
                FEM_SVD_Generic* fem = new FEM_SVD_Generic(ids.data(), get_svd_materials(_material, _young, _poisson), get_fem_shape(type));
                s_fem->add_fem(fem);
                //fems.push_back(fem);
                scalar volume = fem->get_volume();
                for (int j = 0; j < nb; ++j) {
                    Particle* p = s_fem->get(ids[j]);
                    p->mass += _density * volume / nb;
                }
            }
        }
        scalar total_mass = 0;
        for (Particle* p : s_fem->particles()) {
            p->inv_mass = scalar(1) / p->mass;
            total_mass += p->mass;
        }
        std::cout << "FEM TOTAL MASS = " << total_mass << std::endl;
    }


    std::vector<FEM_Generic*> fems;
protected:
    scalar _density;
    int nb_step;
    scalar _young, _poisson;
    int _sub_iteration;
    Material _material;
};