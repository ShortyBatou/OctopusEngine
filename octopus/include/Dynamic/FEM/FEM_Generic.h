#pragma once
#include "Core/Base.h"
#include "Dynamic/Base/Particle.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include <vector>

struct FEM_Generic {
    FEM_Generic(ContinuousMaterial *material, FEM_Shape *shape)
        : _shape(shape), _material(material), _init_volume(0) {
    }

    ~FEM_Generic();

    [[nodiscard]] FEM_Shape *get_shape() const { return _shape; }
    [[nodiscard]] ContinuousMaterial *get_material() const { return _material; }
    [[nodiscard]] scalar get_init_volume() const { return _init_volume; }

    static Matrix3x3 get_jacobian(const std::vector<Particle *> &p, const std::vector<Vector3> &dN);

    static Matrix3x3 get_jacobian(const std::vector<Vector3> &p, const std::vector<Vector3> &dN);

    static scalar compute_volume(const FEM_Shape *shape, const std::vector<Particle *> &p, const Mesh::Topology &topology);
    static scalar compute_volume(const FEM_Shape *shape, const Mesh::Geometry &geometry, const Mesh::Topology &topology);

    [[nodiscard]] scalar compute_volume(const std::vector<Particle *> &p) const;

    [[nodiscard]] scalar compute_volume(const std::vector<Vector3> &p) const;

    [[nodiscard]] virtual scalar compute_stress(const std::vector<Vector3> &p) const;


    // particles : element's particles only and in the right order
    void build(const std::vector<Particle *> &p);

protected:
    FEM_Shape *_shape;
    ContinuousMaterial *_material;
    scalar _init_volume;
    std::vector<Matrix3x3> _JX_inv;
    std::vector<scalar> _V;
};
