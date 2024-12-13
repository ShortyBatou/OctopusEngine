#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"
#include "Dynamic/PBD/PositionBasedDynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"

class XPBD_FEM_Dynamic : public FEM_Dynamic_Generic {
public:
    XPBD_FEM_Dynamic(const scalar density, Mass_Distribution m_distrib,
                     const scalar young, const scalar poisson,
                     const Material material,
                     const int iteration = 1, const int sub_iteration = 30,
                     const scalar global_damping = 0, const bool coupled = false,
                     const PBDSolverType type = GaussSeidel)
    : FEM_Dynamic_Generic(density, m_distrib, young, poisson, material, sub_iteration),
    _global_damping(global_damping), _iteration(iteration), _type(type), _coupled_fem(coupled) {
    }

    [[nodiscard]] int get_iteration() const { return _iteration; }
    [[nodiscard]] int get_sub_iteration() const { return _sub_iteration; }

    void set_iterations(int it, int sub_it);

protected:
    ParticleSystem *build_particle_system() override;

    std::vector<FEM_Generic *> build_element(const std::vector<int> &ids, Element type, scalar &volume) override;

public:
    bool _coupled_fem;
    scalar _global_damping;
    int _iteration;
    PBDSolverType _type;
};
