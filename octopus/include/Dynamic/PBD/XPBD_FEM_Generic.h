#pragma once
#include "Dynamic/PBD/XPBD_Constraint.h"
#include "Dynamic/PBD/PBD_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Generic.h"
#include "Dynamic/FEM/FEM_Shape.h"

class XPBD_FEM_Generic : public XPBD_Constraint, public FEM_Generic {
public:
    XPBD_FEM_Generic(const std::vector<int>& ids, PBD_ContinuousMaterial* material, FEM_Shape* shape)
        : XPBD_Constraint(ids, material->get_stiffness()), FEM_Generic(material, shape), _pbd_material(material)
    { }

    void init(const std::vector<Particle*>& particles) override;
    bool project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) override;

protected:
    PBD_ContinuousMaterial* _pbd_material;
};

class XPBD_FEM_Generic_Coupled final : public XPBD_FEM_Generic {
public:
    XPBD_FEM_Generic_Coupled(const std::vector<int>& ids, const std::vector<PBD_ContinuousMaterial*>& materials, FEM_Shape* shape)
        : XPBD_FEM_Generic(ids, materials.back(), shape), _pbd_materials(materials)
    { }

    void init(const std::vector<Particle*>& particles) override;

    // overide the XPBD solve function to work with coupled FEM constraint
    void apply(const std::vector<Particle*>& particles, scalar dt) override;

    [[nodiscard]] scalar compute_stress(const std::vector<Vector3> &p) const override;

private:
    std::vector<PBD_ContinuousMaterial*> _pbd_materials;
};