#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"
#include <map>
#include <set>

class GL_GraphicSurface : public GL_Graphic {
public:
    using Map_Line = std::set<Face<2> >;
    using Map_Triangle = std::set<Face<3> >;
    using Map_Quad = std::set<Face<4> >;

    explicit GL_GraphicSurface(const Color &color = Color(0.8, 0.4, 0.4, 1.0));

    // find the surface of the mesh (pretty much brute force maybe, there is a better way)
    void update_gl_topology() override;

    void find_surface(std::map<Element, Map_Triangle> &elem_surface_tri, std::map<Element, Map_Quad> &elem_surface_quad,
                      bool use_ref_geometry = false);

    void get_quad_wireframe(std::map<Element, Map_Quad> &elem_surface_quad,
                            std::map<Element, Map_Line> &elem_surface_line) const;

    void get_wireframe_all(
        std::map<Element, Map_Triangle> &elem_surface_tri,
        std::map<Element, Map_Quad> &elem_surface_quad,
        std::map<Element, Map_Line> &elem_surface_line,
        bool high_order_only = true
    );

    void extract_topology_in_gl(
        std::map<Element, Map_Triangle> &elem_surface_tri,
        std::map<Element, Map_Quad> &elem_surface_quad,
        std::map<Element, Map_Line> &elem_surface_line,
        bool high_order_only = true
    ) const;

protected:
    std::map<Element, MeshConverter *> _converters;
};
