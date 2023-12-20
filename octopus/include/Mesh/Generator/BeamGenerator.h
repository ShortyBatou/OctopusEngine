#pragma once
#include "Mesh/Generator/MeshGenerator.h"

struct BeamMeshGenerator : public MeshGenerator
{
    BeamMeshGenerator(const Vector3I& subdivisions, const Vector3& sizes)
        : _subdivisions(subdivisions + Vector3I(1)), _sizes(sizes)
    { 
        _x_step = _sizes.x / scalar(_subdivisions.x - 1);
        _y_step = _sizes.y / scalar(_subdivisions.y - 1);
        _z_step = _sizes.z / scalar(_subdivisions.z - 1);
    }
    

    virtual Mesh* build() override
    {
        Mesh* mesh = new Mesh();
        buildGeometryGrid(mesh->geometry());
        apply_transform(mesh->geometry());
        unsigned int ids[8];
        for (unsigned int z = 0; z < _subdivisions.z - 1; ++z)
        for (unsigned int y = 0; y < _subdivisions.y - 1; ++y)
        for (unsigned int x = 0; x < _subdivisions.x - 1; ++x)
        {
            this->get_cell_vertices_ids(x, y, z, ids);
            addGeometryAtCell(x, y, z, mesh->geometry());
            buildTopoAtCell(ids, mesh->topologies());
        }
        return mesh;
    }

    void buildGeometryGrid(Mesh::Geometry& geometry)
    {
        for (unsigned int z = 0; z < _subdivisions.z; ++z)
        for (unsigned int y = 0; y < _subdivisions.y; ++y)
        for (unsigned int x = 0; x < _subdivisions.x; ++x)
            geometry.push_back(Vector3(x * _x_step, y * _y_step, z * _z_step));
    }

    void get_cell_vertices_ids(unsigned int x, unsigned int y, unsigned int z,
                               unsigned int* ids)
    {
        ids[0] = icoord_to_id(x, y, z);
        ids[1] = icoord_to_id(x + 1, y, z);
        ids[2] = icoord_to_id(x + 1, y, z + 1);
        ids[3] = icoord_to_id(x, y , z+ 1);

        ids[4] = icoord_to_id(x, y + 1, z);
        ids[5] = icoord_to_id(x + 1, y + 1, z);
        ids[6] = icoord_to_id(x + 1, y + 1, z + 1);
        ids[7] = icoord_to_id(x, y + 1, z + 1);
    }

    unsigned int icoord_to_id(unsigned int x, unsigned int y, unsigned int z)
    {
        return x + y * _subdivisions.x + z * _subdivisions.y * _subdivisions.x;
    }

    virtual void addGeometryAtCell(unsigned int x, unsigned int y, unsigned int z, Mesh::Geometry& geometry) { }
    virtual void buildTopoAtCell(unsigned int ids[8], std::map<Element, Mesh::Topology>& topologies) = 0;

protected:
    Vector3I _subdivisions;
    Vector3 _sizes;
    scalar _x_step, _y_step, _z_step;
};

class HexaBeamGenerator : public BeamMeshGenerator
{
public:

    HexaBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes) 
        : BeamMeshGenerator(_subdivisions, _sizes)
    { }

    virtual void buildTopoAtCell(unsigned int ids[8], std::map<Element, Mesh::Topology>& topologies) override
    {
        for (unsigned i = 0; i < 8; ++i) 
            topologies[Hexa].push_back(ids[i]);
    }

    virtual ~HexaBeamGenerator() { }
};

class PrysmBeamGenerator : public BeamMeshGenerator
{
public:

    PrysmBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes) 
        : BeamMeshGenerator(_subdivisions, _sizes)
    { }

    virtual void buildTopoAtCell(unsigned int ids[8], std::map<Element, Mesh::Topology>& topologies)
    {
        static unsigned prysms[12] {0, 1, 3, 4, 5, 7, 
                                    1, 2, 3, 5, 6, 7 
        };
        for (unsigned int i = 0; i < 12; ++i)
            topologies[Prysm].push_back(ids[prysms[i]]);
    }

    virtual ~PrysmBeamGenerator() { }
};

class PyramidBeamGenerator : public BeamMeshGenerator
{
public:
    unsigned int _mid_id;

    PyramidBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes)
        : BeamMeshGenerator(_subdivisions, _sizes), _mid_id(0)
    { }

    void addGeometryAtCell(unsigned int x, unsigned int y, unsigned int z, Mesh::Geometry& geometry) override
    {
        _mid_id = geometry.size();
        Vector3 v(scalar(x * this->_x_step + this->_x_step * 0.5),
                  scalar(y * this->_y_step + this->_y_step * 0.5),
                  scalar(z * this->_z_step + this->_z_step * 0.5));
        geometry.push_back(v);
    }

    virtual void buildTopoAtCell(unsigned int ids[8], std::map<Element, Mesh::Topology>& topologies) override
    {
        static unsigned pyramids[30] {3, 2, 1, 0, 8,
                                      0, 1, 5, 4, 8, 
                                      1, 2, 6, 5, 8, 
                                      2, 3, 7, 6, 8,
                                      3, 0, 4, 7, 8,
                                      4, 5, 6, 7, 8};
        for (unsigned int i = 0; i < 30; ++i)
        {
            unsigned int id;
            if (pyramids[i] == 8) id = _mid_id;
            else id = ids[pyramids[i]];
            topologies[Pyramid].push_back(id);
        }
    }
    virtual ~PyramidBeamGenerator() { }
};


class TetraBeamGenerator : public BeamMeshGenerator
{
public:

    TetraBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes)
        : BeamMeshGenerator(_subdivisions, _sizes)
    { }

    virtual void buildTopoAtCell(unsigned int ids[8], std::map<Element, Mesh::Topology>& topologies) override
    {
        static unsigned tetras[24]{ 0,3,1,4, 4,1,5,3, 3,7,5,4, 1,2,5,3, 5,2,7,3, 5,7,2,6 }; int nb = 24;
        //static unsigned tetras[20]{ 1,6,5,4, 1,2,6,3, 0,1,4,3, 7,6,3,4, 1,3,6,4 }; int nb = 20;
        for (unsigned int i = 0; i < nb; ++i)
            topologies[Tetra].push_back(ids[tetras[i]]);
    }
    virtual ~TetraBeamGenerator() { }
};

void tetra4_to_tetra10(Mesh::Geometry& geometry, std::map<Element, Mesh::Topology>& topologies)
{
    Mesh::Topology tetras = topologies[Tetra];
    topologies[Tetra].clear();
    topologies[Tetra10].clear();

    using Edge = std::pair<unsigned int, unsigned int>;
    unsigned int tetra_10_topo[32] = { 0,4,6,7, 1,5,4,8, 7,8,9,3, 2,6,5,9, 6,4,5,7, 7,4,5,8, 6,5,9,7, 7,8,5,9 };
    Edge tetra_edges[6] = { Edge(0,1), Edge(1,2), Edge(0,2), Edge(0,3), Edge(1,3), Edge(2,3) };
    unsigned int e1, e2;
    Edge e;
    std::map<Edge, unsigned int> edges;
    for (unsigned int i = 0; i < tetras.size(); i += 4) {
        unsigned int ids[10];
        unsigned int j = 0;
        for (; j < 4; ++j) ids[j] = tetras[i + j];

        for (Edge& tet_e : tetra_edges) {
            e1 = ids[tet_e.first]; e2 = ids[tet_e.second];
            if (e1 > e2) std::swap(e1, e2);
            e.first = e1; e.second = e2;

            unsigned int id;
            // edge found in map
            if (edges.find(e) != edges.end()) {
                id = edges[e];
            }
            else {
                id = geometry.size();
                Vector3 pa = Vector3(geometry[e1]);
                Vector3 pb = Vector3(geometry[e2]);

                Vector3 p = scalar(0.5) * (pa + pb);
                geometry.push_back(p);
                edges[e] = id;
            }
            ids[j] = id;
            ++j;
        }

        //for (unsigned int k = 0; k < 32; ++k) {
        //    topologies[Tetra].push_back(ids[tetra_10_topo[k]]);
        //}
        for (unsigned int k = 0; k < 10; ++k) {
            topologies[Tetra10].push_back(ids[k]);
        }

    }
}
