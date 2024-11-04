#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Generic.h"
#include "Dynamic/FEM/FEM_Shape.h"

class XPBD_FEM_Generic final : public XPBD_Constraint, public FEM_Generic {
public:
    XPBD_FEM_Generic(const std::vector<int>& ids, PBD_ContinuousMaterial* material, FEM_Shape* shape)
        : XPBD_Constraint(ids, material->get_stiffness()), FEM_Generic(material, shape), _pbd_material(material)
    { }

    void init(const std::vector<Particle*>& particles) override;
    bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override;

private: 
    PBD_ContinuousMaterial* _pbd_material;
};