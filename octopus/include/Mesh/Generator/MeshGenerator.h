#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"

struct MeshGenerator
{
    MeshGenerator() : _t(Matrix::Identity4x4()) { }
    virtual Mesh* build() = 0;
    virtual void setTransform(const Matrix4x4& t) { _t = t; }

protected:
    virtual void apply_transform(Mesh::Geometry& geometry) { 
        for (unsigned int i = 0; i < geometry.size(); ++i)
        {
            Vector4 p = _t * Vector4(geometry[i].x, geometry[i].y, geometry[i].z, 1.);
            p /= p.w;
            geometry[i] = Vector3(p.x, p.y, p.z);
        }
    }
    Matrix4x4 _t;
};