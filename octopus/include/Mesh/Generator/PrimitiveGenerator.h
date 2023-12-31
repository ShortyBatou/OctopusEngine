#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include "Mesh/Generator/MeshGenerator.h"

struct TriangleMesh : public MeshGenerator {
    TriangleMesh(const Vector3& a, const Vector3& b, const Vector3& c) : _a(a), _b(b), _c(c)
    { }

    virtual Mesh* build() override { 
        Mesh* mesh = new Mesh();
        Mesh::Geometry geometry = {_a,_b,_c};
        Mesh::Topology topology = {0,1,2};
        apply_transform(geometry);
        mesh->setGeometry(geometry);
        mesh->setTopology(Triangle, topology);
        return mesh;
    }

protected:
    Vector3 _a, _b, _c;
};

struct BoxMesh : public MeshGenerator
{
    BoxMesh(scalar sx, scalar sy, scalar sz)
    { 
        _min = Vector3(sx, sy, sz) * -0.5f;
        _max = Vector3(sx, sy, sz) * 0.5f;
    }

    BoxMesh(const Vector3& p_min, const Vector3& p_max) : _min(p_min), _max(p_max) { }

    virtual Mesh* build() override
    {
        Mesh* mesh              = new Mesh();
        Mesh::Geometry geometry;
        Mesh::Topology topology;
        buildGeometry(geometry);
        buildTopology(topology);
        apply_transform(geometry);
        mesh->setGeometry(geometry);
        mesh->setTopology(Triangle, topology);
        return mesh;
    }

    virtual void buildGeometry(Mesh::Geometry& geometry)
    {
        // use binary propreties to describe cube's corners
        //  0 = 000 => 1 = 001 =>  2 =010 => ... => 7 = 111
        for (int i = 0; i < 8; ++i)
        {
            Vector3 v(_min);
            if (i & 1) v.x = _max.x;
            if (i & 2) v.y = _max.y;
            if (i & 4) v.z = _max.z;
            geometry.push_back(v);
        }
    }

    virtual void buildTopology(Mesh::Topology& topology)
    {
        unsigned int topo[24] {0, 1, 2, 3, 1, 5, 3, 7, 4, 5, 0, 1,
                               4, 0, 6, 2, 2, 3, 6, 7, 6, 7, 4, 5};

        for (unsigned int i = 0; i < 24; i+=4)
        {
            topology.push_back(topo[i]);
            topology.push_back(topo[i+1]);
            topology.push_back(topo[i+2]);

            topology.push_back(topo[i+1]);
            topology.push_back(topo[i+2]);
            topology.push_back(topo[i+3]);
        }
    }


protected:
    Vector3 _min, _max;
};


