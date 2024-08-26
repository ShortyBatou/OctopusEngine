#pragma once
#include "Core/Base.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Dynamic/Base/ParticleSystem.h"

class ParticleSystemDynamic : public Component {
public:
    explicit ParticleSystemDynamic(const scalar particle_mass) : _particle_mass(particle_mass), _mesh(nullptr), _ps(nullptr) {
    }

    void init() override;

    virtual void update_mesh();

    virtual ParticleSystem *getParticleSystem() { return _ps; }

    ~ParticleSystemDynamic() override { delete _ps; }

protected:
    virtual void build_particles();

    virtual ParticleSystem *build_particle_system() = 0;

    virtual void build_dynamic() = 0;

    Mesh *_mesh;
    scalar _particle_mass;
    ParticleSystem *_ps;
};
