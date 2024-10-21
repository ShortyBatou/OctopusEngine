#include "Rendering/GL_GraphicSurface.h"
#include <unordered_set>
#include "Mesh/MeshTools.h"

GL_GraphicSurface::GL_GraphicSurface(const Color &color) : GL_Graphic(color) {
    _converters[Tetra] = new TetraConverter();
    _converters[Pyramid] = new PyramidConverter();
    _converters[Prism] = new PrysmConverter();
    _converters[Hexa] = new HexaConverter();
    _converters[Tetra10] = new Tetra10Converter();
    _converters[Tetra20] = new Tetra20Converter();
    _converters[Hexa27] = new Hexa27Converter();
    for (auto &elem: _converters) elem.second->init();
}

// find the surface of the mesh (pretty much brute force maybe, there is a better way)
void GL_GraphicSurface::update_gl_topology() {
    std::map<Element, Map_Line> elem_surface_line;
    std::map<Element, Map_Triangle> elem_surface_tri;
    std::map<Element, Map_Quad> elem_surface_quad;

    // Find mesh surface
    find_surface(elem_surface_tri, elem_surface_quad, false);

    // Get wireframe of quads (only for linear elements)
    get_quad_wireframe(elem_surface_quad, elem_surface_line);

    // Get wireframe of high-order elements
    get_wireframe_all(elem_surface_tri, elem_surface_quad, elem_surface_line, true);

    // Build final topology
    extract_topology_in_gl(elem_surface_tri, elem_surface_quad, elem_surface_line);
}

void GL_GraphicSurface::find_surface(std::map<Element, Map_Triangle> &elem_surface_tri,
                                     std::map<Element, Map_Quad> &elem_surface_quad, const bool use_ref_geometry) {
    for (auto &it: _mesh->topologies()) {
        Element element = it.first;
        if (_converters.find(element) == _converters.end()) continue;

        Mesh::Topology triangles;
        Mesh::Topology quads;

        // convert all elements into triangles and quads
        _converters[element]->convert_element(_mesh->topology(element), triangles, quads);

        // get surface faces
        const Mesh::Topology &ref_topology_tri = _converters[element]->topo_triangle();
        const Mesh::Topology &ref_topology_quad = _converters[element]->topo_quad();
        Map_Triangle surface_tri;
        Map_Quad surface_quad;
        if (!use_ref_geometry) {
            MeshTools::GetSurface<3>(surface_tri, triangles, ref_topology_tri);
            MeshTools::GetSurface<4>(surface_quad, quads, ref_topology_quad);
        } else {
            const Mesh::Geometry &ref_geometry = _converters[element]->geo_ref();
            MeshTools::GetSurface<3>(surface_tri, triangles, ref_topology_tri, ref_geometry);
            MeshTools::GetSurface<4>(surface_quad, quads, ref_topology_quad, ref_geometry);
        }

        // compare to all previous surfaces and remove duplicates
        for (auto &it2: elem_surface_tri) {
            MeshTools::RemoveDuplicates<3>(surface_tri, it2.second);
        }

        for (auto &it2: elem_surface_quad) {
            MeshTools::RemoveDuplicates<4>(surface_quad, it2.second);
        }
        elem_surface_tri[element] = surface_tri;
        elem_surface_quad[element] = surface_quad;
    }
}

void GL_GraphicSurface::get_quad_wireframe(
    std::map<Element, Map_Quad> &elem_surface_quad,
    std::map<Element, Map_Line> &elem_surface_line
) const {
    static int quad_lines[8] = {0, 1, 1, 2, 2, 3, 3, 0};
    //static int tri_lines[6] = { 0,1,1,2,2,0 };
    // get surface quad wireframe for linear elements
    for (auto &it: _mesh->topologies()) {
        Element element = it.first;
        const Map_Quad &surface_quad = elem_surface_quad[element];
        if (surface_quad.empty()) continue;

        // get quads line and remove duplicates
        Map_Line surface_line;
        if (!is_high_order(element)) {
            for (const Face<4> &quad: surface_quad) {
                for (int j = 0; j < 8; j += 2) {
                    int a = quad.ids[quad_lines[j]];
                    int b = quad.ids[quad_lines[j + 1]];
                    Face<2> edge({a, b});
                    auto it2 = surface_line.find(edge);
                    if (it2 == surface_line.end()) {
                        surface_line.insert(edge);
                    }
                }
            }
        }

        // compare to all previous surfaces and remove duplicates
        for (auto &it2: elem_surface_line) {
            MeshTools::RemoveDuplicates<2>(surface_line, it2.second);
        }
        elem_surface_line[element] = surface_line;
    }
}

void GL_GraphicSurface::get_wireframe_all(
    std::map<Element, Map_Triangle> &elem_surface_tri,
    std::map<Element, Map_Quad> &elem_surface_quad,
    std::map<Element, Map_Line> &elem_surface_line,
    bool high_order_only
) {
    static int quad_lines[8] = {0, 1, 1, 2, 2, 3, 3, 0};
    static int tri_lines[6] = {0, 1, 1, 2, 2, 0};
    // build surface wireframe for high-order element
    for (auto &it: _mesh->topologies()) {
        Element element = it.first;
        if (high_order_only && !is_high_order(element)) continue;

        // build wireframe
        // For each element on surface generate its wireframe
        Map_Line element_wireframe;
        std::set<int> e_ids;
        const Mesh::Topology &elem_edges = _converters[element]->topo_edge();
        const Mesh::Topology &elements = _mesh->topology(element);
        const int nb = elem_nb_vertices(element);

        // doublon tri and quad
        for (const Face<3> &tri: elem_surface_tri[element]) {
            int eid = tri.element_id;
            if (e_ids.find(eid) != e_ids.end()) continue;
            for (int i = 0; i < elem_edges.size(); i += 2) {
                int a = elements[eid * nb + elem_edges[i]];
                int b = elements[eid * nb + elem_edges[i + 1]];
                element_wireframe.insert(Face<2>({a, b}));
            }
        }

        for (const Face<4> &quad: elem_surface_quad[element]) {
            int eid = quad.element_id;
            if (e_ids.find(eid) != e_ids.end()) continue;
            for (int i = 0; i < elem_edges.size(); i += 2) {
                int a = elements[eid * nb + elem_edges[i]];
                int b = elements[eid * nb + elem_edges[i + 1]];
                element_wireframe.insert(Face<2>({a, b}));
            }
        }

        Map_Line surface_wireframe;
        for (const Face<3> &tri: elem_surface_tri[element]) {
            for (int i = 0; i < 6; i += 2) {
                int a = tri.ids[tri_lines[i]];
                int b = tri.ids[tri_lines[i + 1]];
                Face<2> edge({a, b});
                if (element_wireframe.find(edge) == element_wireframe.end()) continue;
                surface_wireframe.insert(edge);
            }
        }

        for (const Face<4> &quad: elem_surface_quad[element]) {
            for (int i = 0; i < 8; i += 2) {
                int a = quad.ids[quad_lines[i]];
                int b = quad.ids[quad_lines[i + 1]];
                Face<2> edge({a, b});
                if (element_wireframe.find(edge) == element_wireframe.end()) continue;
                surface_wireframe.insert(edge);
            }
        }

        // compare to all previous surfaces and remove duplicates
        for (auto &it2: elem_surface_line) {
            MeshTools::RemoveDuplicates<2>(surface_wireframe, it2.second);
        }

        elem_surface_line[element] = surface_wireframe;
    }
}

void GL_GraphicSurface::extract_topology_in_gl(
    std::map<Element, Map_Triangle> &elem_surface_tri,
    std::map<Element, Map_Quad> &elem_surface_quad,
    std::map<Element, Map_Line> &elem_surface_line,
    bool high_order_only
) const {
    int quad_triangle[6] = {0, 1, 3, 3, 1, 2};
    for (auto &it: _gl_topologies) {
        Element element = it.first;
        GL_Topology *gl_topo = it.second;

        MeshTools::ExtractTopo<2>(elem_surface_line[element], gl_topo->lines);

        // get surface quads
        Mesh::Topology quads, quad_to_elem;
        MeshTools::ExtractTopo<4>(elem_surface_quad[element], quads);
        MeshTools::ExtractFaceToElem<4>(elem_surface_quad[element], quad_to_elem);

        // quad are divided into two triangles
        gl_topo->quads.resize(quads.size() / 4 * 6);
        gl_topo->quad_to_elem.resize(quads.size() / 4 * 2);
        for (int i = 0; i < quads.size() / 4; i++) {
            for (int j = 0; j < 6; ++j) {
                gl_topo->quads[i * 6 + j] = quads[i * 4 + quad_triangle[j]];
            }
            gl_topo->quad_to_elem[i * 2] = quad_to_elem[i];
            gl_topo->quad_to_elem[i * 2 + 1] = quad_to_elem[i];
        }

        if (!is_high_order(element) && high_order_only) {
            MeshTools::ExtractTopo<3>(elem_surface_tri[element], gl_topo->triangles);
            MeshTools::ExtractFaceToElem<3>(elem_surface_tri[element], gl_topo->tri_to_elem);
        } else {
            // we don't want automatic wire frame for high-order element triangles
            MeshTools::ExtractTopo<3>(elem_surface_tri[element], gl_topo->quads);
            MeshTools::ExtractFaceToElem<3>(elem_surface_tri[element], gl_topo->quad_to_elem);
        }
    }
}
