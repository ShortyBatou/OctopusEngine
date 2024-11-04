#pragma once
#include "Core/Base.h"
#include "Dynamic/PBD/PositionBasedDynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"

class XPBD_ShapeMatching_Dynamic: public ParticleSystemDynamic {
public:
    XPBD_ShapeMatching_Dynamic(const scalar density,
                     const scalar young, const scalar poisson,
                     const Material material,
                     const int iteration = 1, const int sub_iteration = 30,
                     const scalar global_damping = 0,
                     const PBDSolverType type = GaussSeidel)
    : ParticleSystemDynamic(density),
        _density(density), _young(young), _poisson(poisson), _material(material), _global_damping(global_damping),
        _iteration(iteration), _sub_iteration(sub_iteration), _type(type)
    {}
    void update() override;
protected:
    void build_dynamic() override;
    ParticleSystem *build_particle_system() override;
public:
    scalar _density;
    scalar _young, _poisson;
    Material _material;
    scalar _global_damping;
    int _iteration, _sub_iteration;
    PBDSolverType _type;
};
