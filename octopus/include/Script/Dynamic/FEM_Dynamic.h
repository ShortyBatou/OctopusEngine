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

class FEM_Dynamic : public Component {
public:

    virtual void init() override {
        _mesh = this->_entity->getComponent<Mesh>();
        build_obj_fem(100000, 0.45);
    }

    virtual void update() override {
        Time::Tic();
        _fem->step(Time::Fixed_DeltaTime());
        //std::cout << "FEM : " << Time::Tac() * 1000. << " ms" << std::endl;

        if (Input::Loop(Key::C)) {
            _c_lock->set_active(false);
        }
        else {
            _c_lock->set_active(true);
            glm::mat4 r = _c_lock->rot;
            r = glm::rotate(r, glm::radians(Time::DeltaTime() * 3.14f * 180.f * 0.1f), Unit3D::right());
            _c_lock->rot = r;
        }
        
        for (unsigned int i = 0; i < _mesh->nb_vertices(); ++i) {
            _mesh->geometry()[i] = _fem->get(i)->position;
        }

    }

    FEM_Shape* get_shape(Element type) {
        FEM_Shape* shape;
        switch (type) {
        case Tetra: return new Tetra_4(); break;
        case Pyramid: return new Pyramid_5(); break;
        case Prysm: return new Prysm_6(); break;
        case Hexa: return new Hexa_8(); break;
        default: std::cout << "build_element : element not found " << type << std::endl; return nullptr;
        }
    }

    void add_particles() {
        float total_weight = 10.;
        float node_weight = total_weight / _mesh->nb_vertices();

        std::vector<unsigned int> ids;
        std::vector<unsigned int> ids_end;
        for (unsigned int i = 0; i < _mesh->nb_vertices(); ++i) {
            Vector3 p = Vector3(_mesh->geometry()[i]);
            _fem->add_particle(p, node_weight);
            if (p.x < 0.1) {
                ids.push_back(i);
            }

            if (p.x > 3.95) {
                ids_end.push_back(i);
            }
        }

        RB_Fixation* c_lock = new RB_Fixation(ids);
        _fem->add_constraint(c_lock);

        _c_lock = new RB_Fixation(ids_end);
        _fem->add_constraint(_c_lock);

    }

    void build_obj_fem(scalar young, scalar poisson) {
        _fem = new FEM_System(new EulerSemiExplicit(Vector3(0., -9.81, 0.), 0.999), 80);
        add_particles();
        for (auto& topo : _mesh->topologies()) {
            Element type = topo.first;
            unsigned int nb = element_vertices(type);
            std::vector<unsigned int> ids(nb);
            for (unsigned int i = 0; i < topo.second.size(); i += nb) {
                for (unsigned int j = 0; j < nb; ++j) {
                    ids[j] = topo.second[i + j];
                }
                ContinuousMaterial* material = new NeoHooke(young, poisson);
                _fem->add_fem(new FEM_Generic(ids.data(), material, get_shape(type)));
            }

        }
    }



protected:
    unsigned int _nb_x;
    unsigned int _nb_y;
    Mesh* _mesh;
    FEM_System* _fem;
    RB_Fixation* _c_lock;
};