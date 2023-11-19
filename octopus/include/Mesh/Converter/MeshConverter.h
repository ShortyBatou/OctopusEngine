#pragma once
#include "Mesh/Mesh.h"
#include "Manager/Debug.h"
struct MeshConverter
{
    MeshConverter() { 
        
    }
    void init() {
        _topo_triangle = get_elem_topo_triangle();
        _topo_quad     = get_elem_topo_quad();
    }

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

    virtual void build_scaled_topology(std::map<Element, Mesh::Topology>& mesh_topologies,
                  std::map<Element, Mesh::Topology>& elem_topologies) 
    {
        Element elem          = get_element_type();
        const unsigned int nb = element_vertices(elem);
        const unsigned int nb_elem = mesh_topologies[elem].size() / nb;
        resize_topo(nb_elem, _topo_triangle.size(), elem_topologies[Triangle]);
        resize_topo(nb_elem, _topo_quad.size(), elem_topologies[Quad]);
        for (unsigned int i = 0; i < nb_elem; ++i)
        {
            build_scaled_element_topo(i * nb, i, _topo_triangle, elem_topologies[Triangle]);
            build_scaled_element_topo(i * nb, i, _topo_quad, elem_topologies[Quad]);
        }
        
    }

    virtual void convert_element(std::map<Element, Mesh::Topology>& mesh_topologies, std::map<Element, Mesh::Topology>& elem_topologies)
    {
        Element elem               = get_element_type();
        const unsigned int nb      = element_vertices(elem);
        const unsigned int nb_elem = mesh_topologies[elem].size() / nb;
        resize_topo(nb_elem, _topo_triangle.size(), elem_topologies[Triangle]);
        resize_topo(nb_elem, _topo_quad.size(), elem_topologies[Quad]);
        for (unsigned int i = 0; i < nb_elem; ++i)
        {
            convert_element_topo(i*nb, i, _topo_triangle, mesh_topologies[elem], elem_topologies[Triangle]);
            convert_element_topo(i*nb, i, _topo_quad, mesh_topologies[elem], elem_topologies[Quad]);
        }
    }

    Mesh::Topology& topo_triangle() { return _topo_triangle; }
    Mesh::Topology& topo_quad() { return _topo_quad; }

    virtual Element get_element_type() = 0;
    virtual ~MeshConverter() { }

protected:
    virtual Mesh::Topology get_elem_topo_triangle() = 0;
    virtual Mesh::Topology get_elem_topo_quad()     = 0;
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
    Mesh::Topology _topo_triangle, _topo_quad;
};

struct TetraConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Tetra; }
    virtual Mesh::Topology get_elem_topo_triangle() override
    {
        return {0, 1, 3, 1, 2, 3, 0, 2, 1, 0, 3, 2 };
    }
    virtual Mesh::Topology get_elem_topo_quad() override { return Mesh::Topology(); }
    virtual ~TetraConverter() { }
};

struct PyramidConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Pyramid; }
    virtual Mesh::Topology get_elem_topo_triangle() override
    {
        return {0, 1, 4, 3, 0, 4, 1, 2, 4, 2, 3, 4};
    }
    virtual Mesh::Topology get_elem_topo_quad() override
    { 
        return {3,2,1,0};
    }

    virtual ~PyramidConverter() { }
};

struct PrysmConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Prysm; }

    virtual Mesh::Topology get_elem_topo_triangle() override
    {
        return {0, 1, 2, 3, 5, 4};
    }

    virtual Mesh::Topology get_elem_topo_quad() override
    {
        return {3,4,1,0, 2,5,3,0, 1,4,5,2};
    }

    virtual ~PrysmConverter() { } 
};

struct HexaConverter : public MeshConverter
{
    virtual Element get_element_type() override { return Hexa; }

    virtual Mesh::Topology get_elem_topo_quad() override
    {
        return {4,5,1,0, 0,1,2,3, 5,6,2,1, 7,4,0,3, 7,6,5,4, 3,2,6,7};
    }

    virtual Mesh::Topology get_elem_topo_triangle() override { return Mesh::Topology(); }
    virtual ~HexaConverter() { }
};