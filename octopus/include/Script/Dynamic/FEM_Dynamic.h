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
#include "Dynamic/FEM/ContinousMaterial.h"
#include "Dynamic/FEM/FEM_Generic.h"

class FEM_Dynamic : public ParticleSystemDynamic {
public:
    FEM_Dynamic(scalar total_mass, scalar young, scalar poisson, Material material, unsigned int sub_iteration = 30)
        : ParticleSystemDynamic(total_mass), _young(young), _poisson(poisson), _material(material), _sub_iteration(sub_iteration) {
    }

    virtual void update() override {
        Time::Tic();
        this->_ps->step(Time::Fixed_DeltaTime());
        std::cout << "FEM : " << Time::Tac() * 1000. << " ms" << std::endl;
        
        for (unsigned int i = 0; i < _mesh->nb_vertices(); ++i) {
            _mesh->geometry()[i] = this->_ps->get(i)->position;
        }
    }

    ContinuousMaterial* getMaterial() {
        switch (_material)
        {
        case Hooke: return new M_Hooke(_young, _poisson);
        case StVK: return new M_StVK(_young, _poisson);
        case Neo_Hooke: return new M_NeoHooke(_young, _poisson);
        default:
            std::cout << "Material not found" << std::endl;
            return nullptr;
        }
    }

    FEM_Shape* get_shape(Element type) {
        FEM_Shape* shape;
        switch (type) {
        case Tetra: return new Tetra_4(); break;
        case Pyramid: return new Pyramid_5(); break;
        case Prysm: return new Prysm_6(); break;
        case Hexa: return new Hexa_8(); break;
        case Tetra10: return new Tetra_10(); break;
        default: std::cout << "build_element : element not found " << type << std::endl; return nullptr;
        }
    }

    virtual ParticleSystem* build_particle_system() override {
        return new FEM_System(new EulerSemiExplicit(Vector3(0., -9.81, 0.), 0.999), _sub_iteration);
    }


    virtual void build_dynamic() {
        FEM_System* s_fem= static_cast<FEM_System*>(this->_ps);
        for (auto& topo : _mesh->topologies()) {
            Element type = topo.first;
            unsigned int nb = element_vertices(type);
            std::vector<unsigned int> ids(nb);
            for (unsigned int i = 0; i < topo.second.size(); i += nb) {
                for (unsigned int j = 0; j < nb; ++j) {
                    ids[j] = topo.second[i + j];
                }
                s_fem->add_fem(new FEM_Generic(ids.data(), getMaterial(), get_shape(type)));
            }
        }
    }



protected:
    unsigned int nb_step;
    scalar _young, _poisson;
    unsigned int _sub_iteration;
    Material _material;
};