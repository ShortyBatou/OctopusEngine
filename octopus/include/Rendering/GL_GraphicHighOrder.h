#pragma once
#include "Rendering/GL_GraphicSurface.h"
#include "Mesh/Converter/MeshConverter.h"
#include <map>
#include <set>


class GL_GraphicHighOrder final : public GL_GraphicSurface {
    using Map_Line = std::set<Face<2> >;
    using Map_Triangle = std::set<Face<3> >;
    using Map_Quad = std::set<Face<4> >;

public:
    explicit GL_GraphicHighOrder(int quality, const Color &color = Color(0.8, 0.4, 0.4, 1.0));

    void update_gl_vcolors() override;

    void update_gl_geometry() override;

    void update_gl_topology() override;

    void rebuild_geometry(std::map<Element, Map_Triangle> &elem_surface_tri,
                          std::map<Element, Map_Quad> &elem_surface_quad, std::map<int, int> &map_id);

    void rebuild_wireframe(std::map<Element, Map_Line> &elem_surface_line, std::map<int, int> &map_id) const;

    void refine(std::map<Element, Map_Triangle> &elem_surface_tri, std::map<Element, Map_Quad> &elem_surface_quad,
                std::map<Element, Map_Line> &elem_surface_line);

protected:
    std::vector<Element> _v_to_element_type;
    std::vector<int> _v_to_element;
    Mesh::Geometry _ref_geometry;
    int _quality;
    std::map<Element, MeshConverter *> _converters;
};
