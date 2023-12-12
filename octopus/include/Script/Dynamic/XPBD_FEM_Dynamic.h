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
    XPBD_FEM_Dynamic(scalar total_mass, scalar young, scalar poisson, Material material, unsigned int iteration = 1, unsigned int sub_iteration = 30, PBDSolverType type = GaussSeidel, bool pbd_v1 = false)
        : ParticleSystemDynamic(total_mass), _young(young), _poisson(poisson), _material(material), _iteration(iteration), _sub_iteration(sub_iteration), _type(type), _pbd_v1(pbd_v1){
        mean_cost = 0;
        nb_step = 0;
    }

    virtual void update() {

        scalar total_init_volume = 0;
        scalar total_volume = 0;

        Time::Tic();
        this->_ps->step(Time::Fixed_DeltaTime());
        //scalar error = std::abs(total_init_volume - total_volume) / total_init_volume*scalar(100);
        scalar tac = Time::Tac() * 1000;
        mean_cost += tac;
        nb_step++;
        std::cout << num << " PBD : " << tac << ", " << (mean_cost / nb_step) << " ms" << std::endl;
        
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
            materials.push_back(new Stable_NeoHooke_Second(_young, _poisson));
            materials.push_back(new Stable_NeoHooke_First(_young, _poisson));
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
        return new PBD_System(new EulerSemiExplicit(Vector3(0,-9.81,0.), 1.0), _iteration, _sub_iteration, _type);
    }

    virtual void build_dynamic() {
        PBD_System* _pbd = static_cast<PBD_System*>(this->_ps);
        for (auto& topo : _mesh->topologies()) {
            Element type = topo.first;
            unsigned int nb = element_vertices(type);
            std::vector<unsigned int> ids(nb);
            for (unsigned int i = 0; i < topo.second.size(); i += nb) {
                for (unsigned int j = 0; j < nb; ++j) {
                    ids[j] = topo.second[i + j];
                }
                num = element_vertices(type);

                std::vector<PBD_ContinuousMaterial*> materials;
                getMaterials(materials);

                if (_pbd_v1) {
                    for (PBD_ContinuousMaterial* m : materials) {
                        XPBD_FEM_Generic* fem = new XPBD_FEM_Generic(ids.data(), m, get_shape(type));
                        _pbd->add_xpbd_constraint(fem);
                    }
                }
                else {
                    XPBD_FEM_Generic_V2* fem = new XPBD_FEM_Generic_V2(ids.data(), materials, get_shape(type));
                    _pbd->add_xpbd_constraint(fem);
                }
                
            }

        }
        std::cout << "XPBD FEM BUILDED : VERTICES = " << _pbd->particles().size() << std::endl;
    }

protected:
    unsigned int nb_step;
    scalar mean_cost;
    bool _pbd_v1;
    scalar _young, _poisson;
    unsigned int _iteration, _sub_iteration;
    Material _material;
    PBDSolverType _type;
    unsigned int num;
};