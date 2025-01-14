#include "Mesh/Generator/PrimitiveGenerator.h"
#include "Mesh/Generator/BeamGenerator.h"


Mesh *TriangleMesh::build() {
    Mesh *mesh = new Mesh();
    Mesh::Geometry geometry = {_a, _b, _c};
    const Mesh::Topology topology = {0, 1, 2};
    apply_transform(geometry);
    mesh->set_geometry(geometry);
    mesh->set_topology(Triangle, topology);
    return mesh;
}

Mesh *BoxMesh::build() {
    Mesh *mesh = new Mesh();
    Mesh::Geometry geometry;
    Mesh::Topology topology;
    buildGeometry(geometry);
    buildTopology(topology);
    apply_transform(geometry);
    mesh->set_geometry(geometry);
    mesh->set_topology(Triangle, topology);
    return mesh;
}

void BoxMesh::buildGeometry(Mesh::Geometry &geometry) const {
    // use binary propreties to describe cube's corners
    //  0 = 000 => 1 = 001 =>  2 =010 => ... => 7 = 111
    for (int i = 0; i < 8; ++i) {
        Vector3 v(_min);
        if (i & 1) v.x = _max.x;
        if (i & 2) v.y = _max.y;
        if (i & 4) v.z = _max.z;
        geometry.push_back(v);
    }
}

void BoxMesh::buildTopology(Mesh::Topology &topology) {
    constexpr int topo[24]{
        0, 1, 2, 3, 1, 5, 3, 7, 4, 5, 0, 1,
        4, 0, 6, 2, 2, 3, 6, 7, 6, 7, 4, 5
    };

    for (int i = 0; i < 24; i += 4) {
        topology.push_back(topo[i]);
        topology.push_back(topo[i + 1]);
        topology.push_back(topo[i + 2]);

        topology.push_back(topo[i + 1]);
        topology.push_back(topo[i + 2]);
        topology.push_back(topo[i + 3]);
    }
}
