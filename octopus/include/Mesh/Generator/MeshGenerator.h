#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"

struct MeshGenerator {
    virtual ~MeshGenerator() = default;

    MeshGenerator() : _t(Matrix::Identity4x4()) {
    }

    virtual Mesh *build() = 0;

    virtual void setTransform(const Matrix4x4 &t) { _t = t; }

protected:
    virtual void apply_transform(Mesh::Geometry &geometry);

    Matrix4x4 _t;
};
