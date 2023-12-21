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

class XPBD_FEM_Dynamic : public ParticleSystemDynamic {
public:
    XPBD_FEM_Dynamic(scalar density, scalar young, scalar poisson, Material material, unsigned int iteration = 1, unsigned int sub_iteration = 30, PBDSolverType type = GaussSeidel, bool pbd_v1 = false)
        : ParticleSystemDynamic(density), _young(young), _poisson(poisson), _material(material), _iteration(iteration), _sub_iteration(sub_iteration), _type(type), _pbd_v1(pbd_v1), _density(density) {
        mean_cost = 0;
        mean_error = 0;
        nb_step = 0;
    }

    virtual void late_init() override {
        this->update_mesh();
    }

    virtual void update() {

        scalar total_init_volume = 0;
        scalar total_volume = 0;
        Time::Tic();
        this->_ps->step(Time::Fixed_DeltaTime());
        //scalar error = std::abs(total_init_volume - total_volume) / total_init_volume*scalar(100);
        scalar tac = Time::Tac() * 1000;

        for (XPBD_FEM_Generic* fem : fems) {
            total_volume += fem->volume;
            total_init_volume += fem->init_volume;
        }
        
        for (XPBD_FEM_Generic_V2* fem : fems_v2) {
            total_volume += fem->volume;
            total_init_volume += fem->init_volume;
        }


        for (XPBD_FEM_Tetra* fem : t_fems) {
            total_volume += fem->_V;
            total_init_volume += fem->_V_init;
        }
        scalar error = (total_volume - total_init_volume) / total_init_volume * 100.f;
        mean_error += error;
        mean_cost += tac;
        nb_step++;
        std::cout << num << " PBD : " << tac << ", " << (mean_cost / nb_step) << " ms" << "  error = " << error << " :: " <<  (mean_error / nb_step) << "%" << "   Total =" << total_init_volume << ":" << total_volume << std::endl;
        
        this->_ps->draw_debug_constraints();
        //_pbd->draw_debug_effects();
        //_pbd->draw_debug_particles();
        //_pbd->draw_debug_xpbd();

        this->update_mesh();
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

    void getMaterials(std::vector<PBD_ContinuousMaterial*>& materials) {
        switch (_material)
        {
        case Hooke: 
            materials.push_back(new Hooke_First(_young, _poisson));
            materials.push_back(new Hooke_Second(_young, _poisson));
            break;
        case StVK:
            materials.push_back(new StVK_First(_young, _poisson));
            materials.push_back(new StVK_Second(_young, _poisson));
            break;
        case Neo_Hooke:
            materials.push_back(new Stable_NeoHooke_First(_young, _poisson));
            materials.push_back(new Stable_NeoHooke_Second(_young, _poisson));
            break;
        case Developed_Neohooke:
            materials.push_back(new Developed_Stable_NeoHooke_First(_young, _poisson));
            materials.push_back(new Developed_Stable_NeoHooke_Second(_young, _poisson));
            break;
        default:
            std::cout << "Material not found" << std::endl;
            break;
        }
    }

    virtual ~XPBD_FEM_Dynamic() {
    }

protected:
    virtual ParticleSystem* build_particle_system() override {        
        return new PBD_System(new EulerSemiExplicit(Vector3(0.,-9.81,0.), 0.999), _iteration, _sub_iteration, _type);
    }

    virtual void build_dynamic() {
        PBD_System* _pbd = static_cast<PBD_System*>(this->_ps);
        for (Particle* p : _pbd->particles()) p->mass = 0;

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
                num = elem_nb_vertices(type);

                std::vector<PBD_ContinuousMaterial*> materials;
                scalar volume = 0;
                if (_pbd_v1 && type == Tetra) {
                    if (_material == Developed_Neohooke) {
                        materials.push_back(new C_Developed_Stable_NeoHooke_First(_young, _poisson));
                        materials.push_back(new C_Developed_Stable_NeoHooke_Second(_young, _poisson));
                    }
                    else {
                        materials.push_back(new C_Stable_NeoHooke_First(_young, _poisson));
                        materials.push_back(new C_Stable_NeoHooke_Second(_young, _poisson));
                    }
                    
                    for (PBD_ContinuousMaterial* m : materials) {
                        XPBD_FEM_Tetra* fem = new XPBD_FEM_Tetra(ids.data(), m);
                        _pbd->add_xpbd_constraint(fem);
                        t_fems.push_back(fem);
                        volume += fem->get_volume();
                    }
                }
                else {
                    getMaterials(materials);
                    for (PBD_ContinuousMaterial* m : materials) {
                        XPBD_FEM_Generic* fem = new XPBD_FEM_Generic(ids.data(), m, get_shape(type));
                        fems.push_back(fem);
                        _pbd->add_xpbd_constraint(fem);
                        volume += fem->volume;
                    }
                }
                //else {
                //    getMaterials(materials);
                //    XPBD_FEM_Generic_V2* fem = new XPBD_FEM_Generic_V2(ids.data(), materials, get_shape(type));
                //    _pbd->add_xpbd_constraint(fem);
                //    fems_v2.push_back(fem);
                //    volume += fem->volume;
                //}
                
                volume /= materials.size();

                for (unsigned int j = 0; j < nb; ++j) {
                    Particle* p = _pbd->get(ids[j]);
                    p->mass += _density * volume / nb;
                }

            }
        }
        scalar total_mass = 0;
        for (Particle* p : _pbd->particles()) {
            p->inv_mass = scalar(1) / p->mass;
            total_mass += p->mass;
        }
        std::cout << "XPBD FEM BUILDED : VERTICES = " << _pbd->particles().size() << "   ELEMENTS = " << nb_element << "   TOTAL MASS = " << total_mass << std::endl;
    }

public:
    scalar mean_error;
    std::vector< XPBD_FEM_Generic*> fems;
    std::vector< XPBD_FEM_Generic_V2*> fems_v2;
    std::vector< XPBD_FEM_Tetra*> t_fems;
    unsigned int nb_step;
    scalar mean_cost;
    bool _pbd_v1;

protected:
    scalar _density;
    scalar _young, _poisson;
    unsigned int _iteration, _sub_iteration;
    Material _material;
    PBDSolverType _type;
    unsigned int num;
};