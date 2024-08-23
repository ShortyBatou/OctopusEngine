
#pragma once

#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Core/Entity.h"
#include "Mesh/Mesh.h"
#include "Mesh/Elements.h"
#include "Tools/Color.h"
#include "Rendering/GL_Buffer.h"
#include <vector>
#include <map>
#include <numeric>

/// Vertex data for display
struct GL_Geometry {
    GL_Geometry();
    void init_vao() const;
    void clear();
    void load() const;
    void load_colors() const;

    /// GPU BUFFERS
    GL_VAO* vao;                  // vao linked to all array buffer
    GL_Buffer<Vector3>* b_vertex; // array buffer : vertex position  [0]
    GL_Buffer<Color>* b_color;    // array buffer : vertex color     [1]
    GL_Buffer<Vector3>* b_normal; // array buffer : vertex normal    [2]

    /// CPU ARRAYS
    Mesh::Geometry geometry;      // array : vertex position
    std::vector<Color> vcolors;  // array : vertex color
    std::vector<Vector3> v_normals; // array : vertex color
};

/// Topology data for display
struct GL_Topology {
    GL_Topology();

    virtual ~GL_Topology();

    virtual void load();

    virtual void laod_ecolor();

    void clear();

    /// GPU BUFFER
    GL_Buffer<int> *b_line, *b_triangle, *b_quad;
    GL_Buffer<int> *sbo_tri_to_elem, *sbo_quad_to_elem;
    GL_Buffer<Color> *sbo_ecolor;

    /// CPU ARRAY
    Mesh::Topology lines, triangles, quads;
    Mesh::Topology tri_to_elem, quad_to_elem;
    std::vector<Color> ecolors; // color for each element
};

class GL_Graphic : public Component
{
public:
    explicit GL_Graphic(const Color& color = ColorBase::Grey(scalar(0.7)))
        : _color(color), _multi_color(false), _element_color(false), _mesh(nullptr),  _gl_geometry(new GL_Geometry())
    {
    }

    void init() override;

    void late_init() override;

    void update() override;

    void set_ecolors(const Element type, const std::vector<Color>& colors) {
        _gl_topologies[type]->ecolors = colors;
    }

    std::vector<Color>& vcolors() { return _vcolors; }

    void set_vcolors(const std::vector<Color>& colors) {
        _vcolors = colors;
    }

    Color& color() { return _color; }

    [[nodiscard]] bool use_multi_color() const { return _multi_color; }
    void set_multi_color(const bool state) { _multi_color = state; }

    [[nodiscard]] bool use_element_color() const { return _element_color; }
    void set_element_color(bool state) {
        _element_color = state;
    }

    std::map<Element, GL_Topology*> gl_topologies() {return _gl_topologies; };
    [[nodiscard]] GL_Geometry* gl_geometry() const { return _gl_geometry; };
    std::vector<Color> get_vcolor() {
        return _vcolors;
    }
    ~GL_Graphic() override;

    static scalar wireframe_intencity;
    static Color vertice_color;
    static scalar vertice_size;
    static scalar line_size;

protected:

    virtual void update_gl_geometry() {
        _gl_geometry->geometry = _mesh->geometry();
    }

    virtual void update_gl_topology();

    virtual void  update_gl_vcolors() {
        _gl_geometry->vcolors = _vcolors;
    }

    bool _multi_color, _element_color;
    Color _color;

    Mesh* _mesh;
    std::vector<Color> _vcolors;
    GL_Geometry* _gl_geometry;
    std::map<Element, GL_Topology*> _gl_topologies;
};

