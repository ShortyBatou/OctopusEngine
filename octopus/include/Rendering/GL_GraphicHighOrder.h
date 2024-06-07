#pragma once
#include "Rendering/GL_GraphicSurface.h"
#include "Mesh/Converter/MeshConverter.h"
#include "Mesh/MeshTools.h"
#include <map>
#include <set>


class GL_GraphicHighOrder : public GL_GraphicSurface
{
    using Map_Line = std::set<Face<2>>;
    using Map_Triangle = std::set<Face<3>>;
    using Map_Quad = std::set<Face<4>>;
public:
    GL_GraphicHighOrder(int quality, const Color& color = Color(0.8, 0.4, 0.4, 1.0)) : GL_GraphicSurface(color), _quality(quality)
    {
        _converters[Tetra] = new TetraConverter();
        _converters[Pyramid] = new PyramidConverter();
        _converters[Prism] = new PrysmConverter();
        _converters[Hexa] = new HexaConverter();
        _converters[Tetra10] = new Tetra10Converter();
        _converters[Tetra20] = new Tetra20Converter();
        for (auto& elem : _converters) elem.second->init();
    }

    virtual void update_gl_vcolors()
    {
        Element element = Element::Line;
        FEM_Shape* shape = nullptr;
        unsigned int element_size;
        _gl_geometry->vcolors.resize(_ref_geometry.size());
        for (int i = 0; i < _ref_geometry.size(); ++i) {
            if (element != _v_to_element_type[i]) {
                delete shape;
                element = _v_to_element_type[i];
                element_size = elem_nb_vertices(element);
                shape = get_fem_shape(element);
            }
            std::vector<scalar> N_v = shape->build_shape(_ref_geometry[i].x, _ref_geometry[i].y, _ref_geometry[i].z);
            _gl_geometry->vcolors[i] = ColorBase::Black();
            for (unsigned int j = 0; j < N_v.size(); ++j) {
                int id = _mesh->topology(element)[_v_to_element[i] * element_size + j];
                _gl_geometry->vcolors[i] += N_v[j] * _vcolors[id];
            }
        }
        delete shape;
    }

    virtual void update_gl_geometry() override
    { 
        Element element = Element::Line;
        FEM_Shape* shape = nullptr;
        unsigned int element_size;
        _gl_geometry->geometry.resize(_ref_geometry.size());
        for (int i = 0; i < _ref_geometry.size(); ++i) {
            if (element != _v_to_element_type[i]) {
                delete shape;
                element = _v_to_element_type[i];
                element_size = elem_nb_vertices(element);
                shape = get_fem_shape(element);
            }
            std::vector<scalar> N_v = shape->build_shape(_ref_geometry[i].x, _ref_geometry[i].y, _ref_geometry[i].z);
            _gl_geometry->geometry[i] = Unit3D::Zero();
            for (unsigned int j = 0; j < N_v.size(); ++j) {
                int id = _mesh->topology(element)[_v_to_element[i] * element_size + j];
                _gl_geometry->geometry[i] += N_v[j] * _mesh->geometry()[id];
            }
        }
        delete shape;
    }

    virtual void update_gl_topology() override
    {
        std::map < Element, Map_Line> elem_surface_line;
        std::map < Element, Map_Triangle> elem_surface_tri;
        std::map < Element, Map_Quad> elem_surface_quad;

        // Find mesh surface 
        find_surface(elem_surface_tri, elem_surface_quad, true);

        // Get wireframe of high-order elements
        get_wireframe_all(elem_surface_tri, elem_surface_quad, elem_surface_line, false);

        // build the geometry in the reference elements and only keep surface vertices
        std::map<int, int> map_id;
        rebuild_geometry(elem_surface_tri, elem_surface_quad, map_id);

        // rebuild the wireframe with the new vertices
        rebuild_wireframe(elem_surface_line, map_id);

        // refinement
        refine(elem_surface_tri, elem_surface_quad, elem_surface_line);
        
        // Build final topology
        extract_topology_in_gl(elem_surface_tri, elem_surface_quad, elem_surface_line);
    }

    void rebuild_geometry(std::map < Element, Map_Triangle>& elem_surface_tri, std::map < Element, Map_Quad>& elem_surface_quad, std::map<int, int>& map_id) {
        _ref_geometry.clear();
        _v_to_element.clear();
        _v_to_element_type.clear();
        for (const auto& it : _mesh->topologies())
        {
            Element element = it.first;
            int nb = _ref_geometry.size();
            MeshTools::RebuildFaces<3>(elem_surface_tri[element], _ref_geometry, _v_to_element, map_id);
            MeshTools::RebuildFaces<4>(elem_surface_quad[element], _ref_geometry, _v_to_element, map_id);
            _v_to_element_type.insert(_v_to_element_type.end(), _ref_geometry.size() - nb, element);
        }
    }

    void rebuild_wireframe(std::map < Element, Map_Line>& elem_surface_line, std::map<int, int>& map_id) {
        for (const auto& it : _mesh->topologies())
        {
            Element element = it.first;
            std::set<Face<2>> remaped_wireframe;
            for (const Face<2>&edge : elem_surface_line[element]) {
                int a = map_id[edge.ids[0]];
                int b = map_id[edge.ids[1]];
                Face<2> edge2 = (a < b) ? Face<2>({ a, b }) : Face<2>({ b, a });
                remaped_wireframe.insert(edge2);
            }
            elem_surface_line[element] = remaped_wireframe;
        }
    }

    void refine(std::map < Element, Map_Triangle>& elem_surface_tri, std::map < Element, Map_Quad>& elem_surface_quad, std::map < Element, Map_Line>& elem_surface_line) {
        static const Mesh::Topology tri_subdivision_pattern = { 0,3,5, 3,1,4, 3,4,5, 5,4,2 };
        static const Mesh::Topology tri_subdivision_edges = { 0,1, 1,2, 0,2 };
        static const Mesh::Topology quad_subdivision_pattern = { 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,3,6 };
        static const Mesh::Topology quad_subdivision_edges = { 0,1, 1,2, 2,3, 3,0, 4,6 }; // will not work
        for (int i = 0; i < _quality; ++i)
        {
            for (const auto& it : _mesh->topologies())
            {
                std::map<Face<2>, int> edges;
                Element element = it.first;
                int nb = _ref_geometry.size();

                MeshTools::Subdivise<3>(
                    tri_subdivision_pattern,    // how to subdivise face
                    tri_subdivision_edges,      // how to subdivise edges (create new vertices)
                    elem_surface_tri[element],  // surface mesh
                    elem_surface_line[element], // wireframe
                    edges,                      // subdivided edges and their corresponding mid vertice id
                    _ref_geometry,     // the new geometry
                    _v_to_element);    // in which element is defined each vertices

                MeshTools::Subdivise<4>(
                    quad_subdivision_pattern,
                    quad_subdivision_edges,
                    elem_surface_quad[element],
                    elem_surface_line[element],
                    edges,
                    _ref_geometry,
                    _v_to_element);
                _v_to_element_type.insert(_v_to_element_type.end(), _ref_geometry.size() - nb, element);
            }

        }
    }

protected:
    std::vector<Element> _v_to_element_type;
    std::vector<int> _v_to_element;
    Mesh::Geometry _ref_geometry;
    int _quality;
    std::map<Element, MeshConverter*> _converters; 
};
