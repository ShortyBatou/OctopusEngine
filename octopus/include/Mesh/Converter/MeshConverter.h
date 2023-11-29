#pragma once
#include "Mesh/Mesh.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Manager/Debug.h"

struct MeshConverter
{
    MeshConverter() { 
        
    }

    void init() {
        _shape = get_shape();
        _topo_triangle = get_elem_topo_triangle();
        _topo_quad     = get_elem_topo_quad();
        _topo_edge = get_elem_topo_edges();
        _ref_element = get_elem_base_vertices();
    }

    // create for each element a scaled version of its geometry
    virtual void
    build_scaled_geometry(const Mesh::Geometry& mesh_geometry,
                  std::map<Element, Mesh::Topology>& mesh_topologies,
                  Mesh::Geometry& elem_geometry, scalar scale = 0.7)
    {

        Element elem          = get_element_type();
        const unsigned int nb = element_vertices(elem);
        Mesh::Geometry elem_geo(nb);
        Vector3 com;
        for (unsigned int i = 0; i < mesh_topologies[elem].size(); i += nb)
        {
            com = Unit3D::Zero();
            for (unsigned int j = 0; j < nb; ++j)
            {
                elem_geo[j] = mesh_geometry[mesh_topologies[elem][i + j]];
                com += elem_geo[j];
            }
            com /= scalar(nb);
 
            for (unsigned int j = 0; j < nb; ++j)
                elem_geometry.push_back(com + (elem_geo[j] - com) * scale);
                
        }
    }

    // use the geometry scaled version to create the topology of the scaled element
    virtual void build_scaled_topology(std::map<Element, Mesh::Topology>& mesh_topologies, Mesh::Topology& triangles, Mesh::Topology& quads)
    {
        Element elem          = get_element_type();
        const unsigned int nb = element_vertices(elem);
        const unsigned int nb_elem = mesh_topologies[elem].size() / nb;
        resize_topo(nb_elem, _topo_triangle.size(), triangles);
        resize_topo(nb_elem, _topo_quad.size(), quads);
        for (unsigned int i = 0; i < nb_elem; ++i)
        {
            build_scaled_element_topo(i * nb, i, _topo_triangle, triangles);
            build_scaled_element_topo(i * nb, i, _topo_quad, quads);
        }
        
    }

    virtual void build_scaled_topology(std::map<Element, Mesh::Topology>& mesh_topologies, Mesh::Topology& lines, Mesh::Topology& triangles, Mesh::Topology& quads)
    {
        Element elem = get_element_type();
        const unsigned int nb = element_vertices(elem);
        const unsigned int nb_elem = mesh_topologies[elem].size() / nb;
        resize_topo(nb_elem, _topo_edge.size(), lines);
        resize_topo(nb_elem, _topo_triangle.size(), triangles);
        resize_topo(nb_elem, _topo_quad.size(), quads);
        for (unsigned int i = 0; i < nb_elem; ++i)
        {
            build_scaled_element_topo(i * nb, i, _topo_edge, lines);
            build_scaled_element_topo(i * nb, i, _topo_triangle, triangles);
            build_scaled_element_topo(i * nb, i, _topo_quad, quads);
        }

    }


    // convert element into quads and triangles
    virtual void convert_element(std::map<Element, Mesh::Topology>& mesh_topologies, Mesh::Topology& triangles, Mesh::Topology& quads)
    {
        Element elem               = get_element_type();
        const unsigned int nb      = element_vertices(elem);
        const unsigned int nb_elem = mesh_topologies[elem].size() / nb;
        resize_topo(nb_elem, _topo_triangle.size(), triangles);
        resize_topo(nb_elem, _topo_quad.size(), quads);
        for (unsigned int i = 0; i < nb_elem; ++i)
        {
            convert_element_topo(i*nb, i, _topo_triangle, mesh_topologies[elem], triangles);
            convert_element_topo(i*nb, i, _topo_quad, mesh_topologies[elem], quads);
        }
    }



    Mesh::Topology& topo_triangle() { return _topo_triangle; }
    Mesh::Topology& topo_quad() { return _topo_quad; }
    Mesh::Topology& topo_edge() { return _topo_edge; }
    Mesh::Geometry& geo_ref() { return _ref_element; }
    FEM_Shape* shape() { return _shape; }
    virtual Element get_element_type() = 0;
    
    virtual ~MeshConverter() { delete _shape; }

protected:
    virtual FEM_Shape* get_shape() = 0;

    // get the triangles in reference element
    virtual Mesh::Topology get_elem_topo_triangle() = 0;

    // get the quads in reference element
    virtual Mesh::Topology get_elem_topo_quad()     = 0;

    // get the edges in reference element 
    virtual Mesh::Topology get_elem_topo_edges() = 0;

    // get the reference element vertices 
    virtual Mesh::Geometry get_elem_base_vertices() = 0;

    

    virtual void resize_topo(unsigned int nb_elem, unsigned int elem_topo_size, Mesh::Topology& topo) {
        topo.resize(topo.size() + nb_elem * elem_topo_size);
    }

    virtual void build_scaled_element_topo(unsigned int vid, unsigned int num_elem, const Mesh::Topology& elem_topo, Mesh::Topology& topology) {
        const unsigned int size = elem_topo.size();
        for (unsigned int i = 0; i < size; ++i)
            topology[num_elem * size + i] = vid + elem_topo[i];
    }

    virtual void convert_element_topo(unsigned int i_start,
                                      unsigned int num_elem,
                                      const Mesh::Topology& elem_topo ,
                                      const Mesh::Topology& mesh_topologies,
                                      Mesh::Topology& topology)
    {
        const unsigned int size = elem_topo.size();
        for (unsigned int i = 0; i < size; ++i)
            topology[num_elem * size + i]  = mesh_topologies[i_start + elem_topo[i]];    
    }

protected:
    FEM_Shape* _shape;
    Mesh::Topology _topo_triangle, _topo_quad, _topo_edge;
    Mesh::Geometry _ref_element;
};

struct TetraConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Tetra; }

    virtual Mesh::Geometry get_elem_base_vertices() override {
        return { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1) };
    }
    virtual Mesh::Topology get_elem_topo_edges() override {
        return { 0,1, 0,2, 0,3, 1,3, 2,3, 1,2 };
    }
    virtual Mesh::Topology get_elem_topo_triangle() override
    {  return {0, 1, 3, 1, 2, 3, 0, 3, 2, 0, 2, 1}; }
    virtual Mesh::Topology get_elem_topo_quad() override { return Mesh::Topology(); }
    virtual FEM_Shape* get_shape() override { return new Tetra_4(); }
    virtual ~TetraConverter() { }
};

struct PyramidConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Pyramid; }

    virtual Mesh::Geometry get_elem_base_vertices() {
        return { Vector3(-1, 0, -1), Vector3(1, 0, -1), Vector3(1, 0, 1), Vector3(-1, 0, -1), Vector3(0, 1, 0) }; // not sure
    }

    virtual Mesh::Topology get_elem_topo_edges() {
        return { 0,1,1,2,2,3,0,4,1,4,2,4,3,4 };
    }

    virtual Mesh::Topology get_elem_topo_triangle() override
    {
        return {0, 1, 4, 3, 0, 4, 1, 2, 4, 2, 3, 4};
    }
    virtual Mesh::Topology get_elem_topo_quad() override
    { 
        return {3,2,1,0};
    }
    virtual FEM_Shape* get_shape() override { return new Pyramid_5(); }
    virtual ~PyramidConverter() { }
};

struct PrysmConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Prysm; }

    virtual Mesh::Topology get_elem_topo_triangle() override
    {
        return {0, 1, 2, 3, 5, 4};
    }

    virtual Mesh::Geometry get_elem_base_vertices() {
        return { Vector3(0, -1, 0), Vector3(1, -1, 0), Vector3(0, -1, 1), Vector3(0, 1, 0), Vector3(1, 1, 0), Vector3(0, 1, 1) }; // not sure
    }

    virtual Mesh::Topology get_elem_topo_edges() {
        return { 0,1, 1,2, 2,0, 0,3, 1,4, 2,5, 3,4, 4,5, 5,3 };
    }

    virtual Mesh::Topology get_elem_topo_quad() override
    {
        return {3,4,1,0, 2,5,3,0, 1,4,5,2};
    }
    virtual FEM_Shape* get_shape() override { return new Prysm_6(); }
    virtual ~PrysmConverter() { } 
};

struct HexaConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Hexa; }

    virtual Mesh::Geometry get_elem_base_vertices() {
        return { Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(1, 1, -1), Vector3(-1, 1, -1),
                 Vector3(-1, -1,  1), Vector3(1,  -1, 1), Vector3(1, 1, 1), Vector3(-1,  1, 1) 
        };
    }

    virtual Mesh::Topology get_elem_topo_edges() {
        return { 0,1, 1,2, 2,3, 3,0, 0,4, 1,5, 2,6, 3,7, 4,5, 5,6, 6,7, 7,4 }; 
    }

    virtual Mesh::Topology get_elem_topo_quad() override
    {
        return {4,5,1,0, 0,1,2,3, 5,6,2,1, 7,4,0,3, 7,6,5,4, 3,2,6,7};
    }

    virtual Mesh::Topology get_elem_topo_triangle() override { return Mesh::Topology(); }
    virtual FEM_Shape* get_shape() override { return new Hexa_8(); }
    virtual ~HexaConverter() { }
};


struct Tetra10Converter : public TetraConverter
{
    virtual Element get_element_type() override { return Tetra10; }

    virtual Mesh::Geometry get_elem_base_vertices() {
        return { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1),
                 Vector3(0.5, 0, 0), Vector3(0.5, 0.5, 0.), Vector3(0., 0.5, 0.),
                 Vector3(0, 0, 0.5), Vector3(0.5, 0., 0.5),   Vector3(0., 0.5, 0.5)
        };
    }

    virtual Mesh::Topology get_elem_topo_edges() {
        return { 0,4, 4,1, 1,5, 5,2, 0,6, 6,2, 0,7, 7,3, 1,8, 8,3, 2,9, 9,3 };
    }

    virtual Mesh::Topology get_elem_topo_triangle() override { 
        return {
            0,4,7, 4,1,8, 4,8,7, 7,8,3,
            1,5,8, 5,9,8, 5,2,9, 8,9,3,
            2,6,9, 6,0,7, 6,7,9, 9,7,3,
            0,6,4, 6,5,4, 6,2,5, 4,5,1
        };
    }

    virtual Mesh::Topology get_elem_topo_quad() override
    {
        return Mesh::Topology();;
    }

    
    virtual FEM_Shape* get_shape() override { return new Tetra_10(); }
};