#pragma once
#include "Core/Base.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Dynamic/Base/ParticleSystem.h"

struct ParticleSystemDynamics_Getters
{
    virtual ~ParticleSystemDynamics_Getters() = default;
    [[nodiscard]] virtual std::vector<Vector3> get_positions() = 0;
    [[nodiscard]] virtual std::vector<Vector3> get_last_positions() = 0;
    [[nodiscard]] virtual std::vector<Vector3> get_init_positions() = 0;
    [[nodiscard]] virtual std::vector<Vector3> get_displacement() = 0;
    [[nodiscard]] virtual std::vector<Vector3> get_velocity() = 0;
    [[nodiscard]] virtual std::vector<int> get_masks() = 0;
    [[nodiscard]] virtual std::vector<scalar> get_masses() = 0;
    [[nodiscard]] virtual std::vector<scalar> get_massses_inv() = 0;
    [[nodiscard]] virtual std::vector<scalar> get_displacement_norm() = 0;
    [[nodiscard]] virtual std::vector<scalar> get_velocity_norm() = 0;
};

class ParticleSystemDynamic : public Component, public ParticleSystemDynamics_Getters {
public:
    explicit ParticleSystemDynamic(const scalar particle_mass) : _particle_mass(particle_mass), _mesh(nullptr), _ps(nullptr) {
    }

    void init() override;

    virtual void update_mesh();

    virtual ParticleSystem *getParticleSystem() { return _ps; }

    ~ParticleSystemDynamic() override { delete _ps; }

    [[nodiscard]] std::vector<Vector3> get_last_positions() override;
    [[nodiscard]] std::vector<Vector3> get_positions() override;
    [[nodiscard]] std::vector<Vector3> get_init_positions() override;
    [[nodiscard]] std::vector<Vector3> get_displacement() override;
    [[nodiscard]] std::vector<Vector3> get_velocity() override;
    [[nodiscard]] std::vector<int> get_masks() override;
    [[nodiscard]] std::vector<scalar> get_masses() override;
    [[nodiscard]] std::vector<scalar> get_massses_inv() override;
    [[nodiscard]] std::vector<scalar> get_displacement_norm() override;
    [[nodiscard]] std::vector<scalar> get_velocity_norm() override;

protected:
    virtual void build_particles();

    virtual ParticleSystem *build_particle_system() = 0;

    virtual void build_dynamic() = 0;

    Mesh *_mesh;
    scalar _particle_mass;
    ParticleSystem *_ps;
};
