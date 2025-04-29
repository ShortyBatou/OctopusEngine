#include "Mesh/Generator/MeshGenerator.h"

void MeshGenerator::apply_transform(Mesh::Geometry &geometry) {
    for (auto &i: geometry) {
        Vector4 p = _t * Vector4(i.x, i.y, i.z, 1.);
        p /= p.w;
        i = Vector3(p.x, p.y, p.z);
    }
}


void MeshGenerator::apply_transform(Vector3& v) {
    Vector4 p = _t * Vector4(v.x, v.y, v.z, 1.);
    p /= p.w;
    v = Vector3(p.x, p.y, p.z);
}