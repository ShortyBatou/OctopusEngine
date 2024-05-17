#pragma once
#include "Mesh/Mesh.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Manager/Debug.h"

struct MeshConverter
{
    MeshConverter() : _shape(nullptr) {
        
    }

    void init() {
        _shape = get_shape();
        _topo_triangle = get_elem_topo_triangle();
        _topo_quad     = get_elem_topo_quad();
        _topo_edge = get_elem_topo_edges();
        _ref_element = _shape->get_vertices();
    }

    // create for each element a scaled version of its geometry
    virtual void
    build_scaled_geometry(const Mesh::Geometry& mesh_geometry,
                  std::map<Element, Mesh::Topology>& mesh_topologies,
                  Mesh::Geometry& elem_geometry, scalar scale = 0.7)
    {

        Element elem          = get_element_type();
        const int nb = elem_nb_vertices(elem);
        Mesh::Geometry elem_geo(nb);
        Vector3 com;
        for (int i = 0; i < mesh_topologies[elem].size(); i += nb)
        {
            com = Unit3D::Zero();
            for (int j = 0; j < nb; ++j)
            {
                elem_geo[j] = mesh_geometry[mesh_topologies[elem][i + j]];
                com += elem_geo[j];
            }
            com /= scalar(nb);
 
            for (int j = 0; j < nb; ++j)
                elem_geometry.push_back(com + (elem_geo[j] - com) * scale);
                
        }
    }

    virtual void build_scaled_topology(
        Mesh::Topology& mesh_topology, 
        Mesh::Topology& triangles, 
        Mesh::Topology& quads,
        Mesh::Topology & tri_to_elem,
        Mesh::Topology& quad_to_elem)
    {
        Element elem = get_element_type();
        const int nb = elem_nb_vertices(elem);
        const int nb_elem = mesh_topology.size() / nb;
        resize_topo(nb_elem, _topo_triangle.size(), triangles);
        resize_topo(nb_elem, _topo_quad.size(), quads);

        resize_topo(nb_elem, _topo_triangle.size() / 3, tri_to_elem);
        resize_topo(nb_elem, _topo_quad.size() / 4, quad_to_elem);
        for (int i = 0; i < nb_elem; ++i)
        {
            build_scaled_element_topo(i * nb, i, _topo_triangle, triangles);
            build_scaled_element_topo(i * nb, i, _topo_quad, quads);
            for (int j = 0; j < _topo_triangle.size() / 3; ++j) {
                tri_to_elem[i * _topo_triangle.size() / 3 + j] = i;
            }
            for (int j = 0; j < _topo_quad.size() / 4; ++j) {
                quad_to_elem[i * _topo_quad.size() / 4 + j] = i;
            }
        }
    }

    virtual void get_scaled_wireframe(Mesh::Topology& mesh_topology, Mesh::Topology& lines) {
        Element elem = get_element_type();
        const int nb = elem_nb_vertices(elem);
        const int nb_elem = mesh_topology.size() / nb;
        resize_topo(nb_elem, _topo_edge.size(), lines);
        for (int i = 0; i < nb_elem; ++i)
        {
            build_scaled_element_topo(i * nb, i, _topo_edge, lines);
        }
    }


    // convert element into quads and triangles
    virtual void convert_element(
        Mesh::Topology& mesh_topology, 
        Mesh::Topology& triangles, 
        Mesh::Topology& quads, 
        Mesh::Topology& tri_to_elem, 
        Mesh::Topology& quad_to_elem)
    {
        Element elem      = get_element_type();
        const int nb      = elem_nb_vertices(elem);
        const int nb_elem = mesh_topology.size() / nb;
        resize_topo(nb_elem, _topo_triangle.size(), triangles);
        resize_topo(nb_elem, _topo_quad.size(), quads);
        resize_topo(nb_elem, _topo_triangle.size()/3, tri_to_elem);
        resize_topo(nb_elem, _topo_quad.size()/4, quad_to_elem);
        for (int i = 0; i < nb_elem; ++i)
        {
            convert_element_topo(i*nb, i, _topo_triangle, mesh_topology, triangles);
            convert_element_topo(i*nb, i, _topo_quad, mesh_topology, quads);
            for (int j = 0; j < _topo_triangle.size() / 3; ++j) {
                tri_to_elem[i * _topo_triangle.size() / 3 + j] = i;
            }
            for (int j = 0; j < _topo_quad.size() / 4; ++j) {
                quad_to_elem[i * _topo_quad.size() / 4 + j] = i;
            }
        }
    }

    virtual void convert_element_topo(int i_start,
        int num_elem,
        const Mesh::Topology& elem_topo,
        const Mesh::Topology& mesh_topologies,
        Mesh::Topology& topology)
    {
        const int size = elem_topo.size();
        for (int i = 0; i < size; ++i) {
            topology[num_elem * size + i] = mesh_topologies[i_start + elem_topo[i]];
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
    

    virtual void resize_topo(int nb_elem, int elem_topo_size, Mesh::Topology& topo) {
        topo.resize(topo.size() + nb_elem * elem_topo_size);
    }

    virtual void build_scaled_element_topo(int vid, int num_elem, const Mesh::Topology& elem_topo, Mesh::Topology& topology) {
        const int size = elem_topo.size();
        for (int i = 0; i < size; ++i) {
            topology[num_elem * size + i] = vid + elem_topo[i];
        }
    }

protected:
    FEM_Shape* _shape;
    Mesh::Topology _topo_triangle, _topo_quad, _topo_edge;
    Mesh::Geometry _ref_element;
};


struct TetraConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Tetra; }

    virtual Mesh::Topology get_elem_topo_edges() override {
        return { 0,1, 1,2, 2,0, 0,3, 1,3, 2,3 };
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
    virtual Element get_element_type() override { return Prism; }

    virtual Mesh::Topology get_elem_topo_triangle() override
    {
        return {0, 1, 2, 3, 5, 4};
    }


    virtual Mesh::Topology get_elem_topo_edges() {
        return { 0,1, 1,2, 2,0, 0,3, 1,4, 2,5, 3,4, 4,5, 5,3 };
    }

    virtual Mesh::Topology get_elem_topo_quad() override
    {
        return {3,4,1,0, 2,5,3,0, 1,4,5,2};
    }
    virtual FEM_Shape* get_shape() override { return new Prism_6(); }
    virtual ~PrysmConverter() { } 
};

struct HexaConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Hexa; }



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

struct Tetra20Converter : public TetraConverter
{
    virtual Element get_element_type() override { return Tetra20; }


    virtual Mesh::Topology get_elem_topo_edges() {
        return 
        {  
            0,4, 4,5, 5,1, 1,6, 6,7, 7,2, 2,8, 8,9, 9,0, 
            0,10, 10,11, 11,3, 3,15, 15,14, 14,2, 1,12, 12,13, 13,3
        };
    }

    virtual Mesh::Topology get_elem_topo_triangle() override {
        return { 
            10,0,4, 10,4,16, 11,10,16, 11,16,13,  3,11,13, 16,4,5, 16,5,12, 13,16,12, 12,5,1,
            12,1,6, 12,6,17, 13,12,17, 13,17,15, 3,13,15, 17,6,7, 17,7,14, 15,17,14, 14,7,2,
            14,2,8, 14,8,18, 15,14,18, 15,18,11, 3,15,11, 18,8,9, 18,9,10, 11,18,10, 10,9,0,
            4,0,9,  4,9,19,  5,4,19,   5,19,6,   1,5,6,   19,9,8, 19,8,7,  6,19,7,   7,8,2

        };
    }

    virtual Mesh::Topology get_elem_topo_quad() override
    {
        return Mesh::Topology();
    }


    virtual FEM_Shape* get_shape() override { return new Tetra_20(); }
};