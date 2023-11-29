#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"

#include "Dynamic/Base/ParticleSystem.h"
class ParticleSystemDynamic : public Component {
public:
    ParticleSystemDynamic(scalar total_mass) : _total_mass(total_mass) { }

    virtual void init() override {
        _mesh = this->entity()->getComponent<Mesh>();
        _ps = build_particle_system();
        build_particles();
        build_dynamic();
    }

    virtual void update_mesh() {
        for (unsigned int i = 0; i < this->_mesh->nb_vertices(); ++i) {
            _mesh->geometry()[i] = _ps->get(i)->position;
        }
    }

    virtual ParticleSystem* getParticleSystem() {
        return _ps;
    }


    virtual ~ParticleSystemDynamic() {
        delete _ps;
    }
protected:
    void build_particles() {
        scalar node_mass = _total_mass / _mesh->nb_vertices();
        for (unsigned int i = 0; i < _mesh->nb_vertices(); ++i) {
            Vector3 p = Vector3(_mesh->geometry()[i]);
            _ps->add_particle(p, node_mass);
        }
    }

    virtual ParticleSystem* build_particle_system() = 0;
    virtual void build_dynamic() = 0;

protected:
    Mesh* _mesh;
    scalar _total_mass;
    ParticleSystem* _ps;
};