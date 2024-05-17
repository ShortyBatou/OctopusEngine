#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"
#include <map>
#include <set>
#include <unordered_set>

class GL_GraphicSurface : public GL_Graphic
{
public:
    using Map_Line = std::set<Face<2>>;
    using Map_Triangle = std::set<Face<3>>;
    using Map_Quad = std::set<Face<4>>;

    GL_GraphicSurface(const Color& color = Color(0.8, 0.4, 0.4, 1.0)) : GL_Graphic(color)
    {
        _converters[Tetra]   = new TetraConverter();
        _converters[Pyramid] = new PyramidConverter();
        _converters[Prism]   = new PrysmConverter();
        _converters[Hexa]    = new HexaConverter();
        _converters[Tetra10] = new Tetra10Converter();
        _converters[Tetra20] = new Tetra20Converter();
        for (auto& elem : _converters) elem.second->init();
    }

    // find the surface of the mesh (pretty much brute force maybe, there is a better way)
    virtual void update_gl_topology() override
    {
        std::map < Element, Map_Line> elem_surface_line;
        std::map < Element, Map_Triangle> elem_surface_tri;
        std::map < Element, Map_Quad> elem_surface_quad;

        // Find mesh surface 
        for (auto& it : _mesh->topologies()) {
            Element element = it.first;
            if (_converters.find(element) == _converters.end()) continue;

            Mesh::Topology triangles, triangle_to_elem;
            Mesh::Topology quads, quad_to_elem;
            
            // convert all elements into triangles and quads
            _converters[element]->convert_element(_mesh->topology(element), triangles, quads, triangle_to_elem, quad_to_elem);

            // get surface mesh for one type of element
            Map_Triangle surface_tri; Map_Quad surface_quad;
            get_surface<3>(surface_tri, triangles, triangle_to_elem);
            get_surface<4>(surface_quad, quads, quad_to_elem);

            // compare to all previous surfaces and remove duplicates
            for (auto& it : elem_surface_tri) {
                compare_surface<3>(surface_tri, it.second);
            }                

            for (auto& it : elem_surface_quad) {
                compare_surface<4>(surface_quad, it.second);
            }  
            elem_surface_tri[element] = surface_tri;
            elem_surface_quad[element] = surface_quad;
        }
        
        static int quad_lines[8] = { 0,1,1,2,2,3,3,0 };
        static int tri_lines[6] = { 0,1,1,2,2,0 };
        static int quad_triangle[6] = { 0,1,3, 3,1,2 };

        // get surface quad wireframe for linear elements
        for (auto& it : _mesh->topologies()) {
            Element element = it.first;
            Map_Quad& surface_quad = elem_surface_quad[element];
            if (surface_quad.size() == 0) continue;

            // get quads line and remove duplicates
            Map_Line surface_line;
            if (!is_high_order(element)) {
                for (const Face<4>&quad : surface_quad)
                {
                    for (int j = 0; j < 8; j += 2) {
                        int a = quad.ids[quad_lines[j]];
                        int b = quad.ids[quad_lines[j + 1]];
                        Face<2> edge({ a,b });
                        auto it = surface_line.find(edge);
                        if (it == surface_line.end()) {
                            surface_line.insert(edge);
                        }
                    }
                }
            }

            // compare to all previous surfaces and remove duplicates
            for (auto& it : elem_surface_line) {
                compare_surface<2>(surface_line, it.second);
            }
            elem_surface_line[element] = surface_line;
        }

        // build surface wireframe for high-order element
        for (auto& it : _mesh->topologies()) {
            Element element = it.first;
            if (!is_high_order(element)) continue;

            // build wireframe
            // For each element on surface generate its wireframe
            Map_Line element_wireframe;
            std::set<int> e_ids;
            const Mesh::Topology& elem_edges = _converters[element]->topo_edge();
            const Mesh::Topology& elements = _mesh->topology(element);
            const int nb = elem_nb_vertices(element);

            // doublon tri and quad
            for (const Face<3>&tri : elem_surface_tri[element]) {
                int eid = tri.element_id;
                if (e_ids.find(eid) != e_ids.end()) continue;
                for (int i = 0; i < elem_edges.size(); i += 2) {
                    int a = elements[eid * nb + elem_edges[i]];
                    int b = elements[eid * nb + elem_edges[i + 1]];
                    element_wireframe.insert(Face<2>({ a,b }));
                }
            }

            for (const Face<4>&quad : elem_surface_quad[element]) {
                int eid = quad.element_id;
                if (e_ids.find(eid) != e_ids.end()) continue;
                for (int i = 0; i < elem_edges.size(); i += 2) {
                    int a = elements[eid * nb + elem_edges[i]];
                    int b = elements[eid * nb + elem_edges[i + 1]];
                    element_wireframe.insert(Face<2>({ a,b }));
                }
            }

            Map_Line surface_wireframe;
            for (const Face<3>&tri : elem_surface_tri[element]) {
                for (int i = 0; i < 6; i += 2) {
                    int a = tri.ids[tri_lines[i]];
                    int b = tri.ids[tri_lines[i+1]];
                    Face<2> edge({ a, b });
                    if (element_wireframe.find(edge) == element_wireframe.end()) continue;
                    surface_wireframe.insert(edge);
                }
            }

            for (const Face<4>&quad : elem_surface_quad[element]) {
                for (int i = 0; i < 8; i += 2) {
                    int a = quad.ids[quad_lines[i]];
                    int b = quad.ids[quad_lines[i + 1]];
                    Face<2> edge({ a, b });
                    if (element_wireframe.find(edge) == element_wireframe.end()) continue;
                    surface_wireframe.insert(edge);
                }
            }

            // compare to all previous surfaces and remove duplicates
            for (auto& it : elem_surface_line) {
                compare_surface<2>(surface_wireframe, it.second);
            }

            elem_surface_line[element] = surface_wireframe;
        }


        // Build final topology
        for (auto& it : _gl_topologies) 
        {
            Element element = it.first;
            GL_Topology* gl_topo = it.second;

            extract_surface_topo<2>(elem_surface_line[element], gl_topo->lines);

            // quad need to be divided into two triangle
            Mesh::Topology quads, quad_to_elem;
            extract_surface_topo<4>(elem_surface_quad[element], quads, quad_to_elem);

            gl_topo->quads.resize(quads.size() / 4 * 6);
            gl_topo->quad_to_elem.resize(quads.size() / 4 * 2);
            for (int i = 0; i < quads.size() / 4; i++) {
                for (int j = 0; j < 6; ++j) {
                    gl_topo->quads[i * 6 + j] = quads[i * 4 + quad_triangle[j]];
                }
                gl_topo->quad_to_elem[i*2] = quad_to_elem[i];
                gl_topo->quad_to_elem[i*2+1] = quad_to_elem[i];
            }

            if (!is_high_order(element)) {
                extract_surface_topo<3>(elem_surface_tri[element], gl_topo->triangles, gl_topo->tri_to_elem);
            }
            else {
                // we don't want automatic wire frame for high-order element triangles
                extract_surface_topo<3>(elem_surface_tri[element], gl_topo->quads, gl_topo->quad_to_elem);
            }
        }

    }

protected:
    /// convert the face set into an indices array
    template<int nb>
    void extract_surface_topo(const std::set<Face<nb>>& faces, Mesh::Topology& topology) {
        topology.resize(nb * faces.size());
        Mesh::Topology ids(nb);
        int i = 0;
        for (const Face<nb>& face : faces) {
            for (int j = 0; j < nb; ++j) {
                topology[i * nb + j] = face.ids[j];
            }
            i++;
        }
    }

    /// convert the face set into a topology and "face_to_element" arrays
    template<int nb>
    void extract_surface_topo(const std::set<Face<nb>>& faces, Mesh::Topology& topology, Mesh::Topology& face_to_elem) {
        int t_size = topology.size();
        int f_size = face_to_elem.size();
        topology.resize(t_size + nb * faces.size());
        face_to_elem.resize(f_size + faces.size());
        Mesh::Topology ids(nb);
        int i = 0;
        for (const Face<nb>& face : faces) {
            face_to_elem[f_size + i] = face.element_id;
            for (int j = 0; j < nb; ++j) {
                topology[t_size + i * nb + j] = face.ids[j];
            }
            i++;
        }
    }

    /// Build a face set from a topology and "face_to_element" arrays. Removes duplicates
    template<int nb>
    void get_surface(std::set<Face<nb>>& faces, Mesh::Topology& topology, Mesh::Topology& face_to_elem)
    {
        Mesh::Topology ids(nb);
        for (int i = 0; i < topology.size(); i += nb)
        {
            for (int j = 0; j < nb; ++j) ids[j] = topology[i + j];

            Face<nb> face(ids);
            face.element_id = face_to_elem[i / nb];
            auto it = faces.find(face);
            if (it == faces.end()) 
                faces.insert(face);
            else
                faces.erase(it);
        }
    }

    /// removes duplicate between two face sets
    template<int nb>
    void compare_surface(std::set<Face<nb>>& a_faces, std::set<Face<nb>>& b_faces) {
        for (const Face<nb>& face : a_faces) {
            auto it = b_faces.find(face);
            if (it != b_faces.end()) {
                a_faces.erase(face);
                b_faces.erase(it);
            }
        }
    }

protected:
    std::map<Element, MeshConverter*> _converters;
};