#include "Mesh/Converter/MeshConverter.h"

void MeshConverter::init() {
    _shape = get_shape();
    _topo_triangle = get_elem_topo_triangle();
    _topo_quad     = get_elem_topo_quad();
    _topo_edge = get_elem_topo_edges();
    _ref_element = _shape->get_vertices();
}

// create for each element a scaled version of its geometry
void MeshConverter::build_scaled_geometry(const Mesh::Geometry& mesh_geometry,
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

void MeshConverter::build_scaled_topology(
    Mesh::Topology& mesh_topology,
    Mesh::Topology& triangles,
    Mesh::Topology& quads)
{
    Element elem = get_element_type();
    const int nb = elem_nb_vertices(elem);
    const int nb_elem = static_cast<int>(mesh_topology.size()) / nb;
    resize_topo(nb_elem, static_cast<int>(_topo_triangle.size()), triangles);
    resize_topo(nb_elem, static_cast<int>(_topo_quad.size()), quads);

    for (int i = 0; i < nb_elem; ++i)
    {
        build_scaled_element_topo(i * nb, i, _topo_triangle, triangles);
        build_scaled_element_topo(i * nb, i, _topo_quad, quads);
    }
}

void MeshConverter::get_scaled_wireframe(Mesh::Topology& mesh_topology, Mesh::Topology& lines) {
    Element elem = get_element_type();
    const int nb = elem_nb_vertices(elem);
    const int nb_elem = static_cast<int>(mesh_topology.size()) / nb;
    resize_topo(nb_elem, static_cast<int>(_topo_edge.size()), lines);
    for (int i = 0; i < nb_elem; ++i)
    {
        build_scaled_element_topo(i * nb, i, _topo_edge, lines);
    }
}


// convert element into quads and triangles
void MeshConverter::convert_element(
    Mesh::Topology& mesh_topology,
    Mesh::Topology& triangles,
    Mesh::Topology& quads)
{
    Element elem      = get_element_type();
    const int nb      = elem_nb_vertices(elem);
    const int nb_elem = static_cast<int>(mesh_topology.size()) / nb;
    resize_topo(nb_elem, static_cast<int>(_topo_triangle.size()), triangles);
    resize_topo(nb_elem, static_cast<int>(_topo_quad.size()), quads);
    for (int i = 0; i < nb_elem; ++i)
    {
        convert_element_topo(i*nb, i, _topo_triangle, mesh_topology, triangles);
        convert_element_topo(i*nb, i, _topo_quad, mesh_topology, quads);
    }
}

void MeshConverter::convert_element_topo(int i_start,
    int num_elem,
    const Mesh::Topology& elem_topo,
    const Mesh::Topology& mesh_topologies,
    Mesh::Topology& topology)
{
    const int size = static_cast<int>(elem_topo.size());
    for (int i = 0; i < size; ++i) {
        topology[num_elem * size + i] = mesh_topologies[i_start + elem_topo[i]];
    }
}

void MeshConverter::build_scaled_element_topo(const int vid, const int num_elem, const Mesh::Topology& elem_topo, Mesh::Topology& topology) {
    const int size = static_cast<int>(elem_topo.size());
    for (int i = 0; i < size; ++i) {
        topology[num_elem * size + i] = vid + elem_topo[i];
    }
}