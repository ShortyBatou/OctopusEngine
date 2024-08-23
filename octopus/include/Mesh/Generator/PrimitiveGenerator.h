#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include "Mesh/Generator/MeshGenerator.h"

struct TriangleMesh final : MeshGenerator {
    TriangleMesh(const Vector3 &a, const Vector3 &b, const Vector3 &c) : _a(a), _b(b), _c(c) {
    }

    Mesh *build() override;

protected:
    Vector3 _a, _b, _c;
};

struct BoxMesh final : MeshGenerator {
    BoxMesh(const scalar sx, const scalar sy, const scalar sz) {
        _min = Vector3(sx, sy, sz) * -0.5f;
        _max = Vector3(sx, sy, sz) * 0.5f;
    }

    BoxMesh(const Vector3 &p_min, const Vector3 &p_max) : _min(p_min), _max(p_max) {
    }

    Mesh *build() override;

    void buildGeometry(Mesh::Geometry &geometry) const;

    void buildTopology(Mesh::Topology &topology);

protected:
    Vector3 _min{}, _max{};
};
