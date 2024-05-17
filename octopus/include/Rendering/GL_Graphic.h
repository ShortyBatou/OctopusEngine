#pragma once

#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Core/Entity.h"
#include "Mesh/Mesh.h"
#include "Mesh/Elements.h"
#include "Tools/Color.h"
#include "Rendering/GL_Buffer.h"
#include "gl_base.h"
#include <vector>
#include <map>
#include <numeric>

/// Vertex data for display
struct GL_Geometry {
    GL_Geometry() {
        vao = new GL_VAO();
        b_vertex = new GL_Buffer<Vector3>(GL_ARRAY_BUFFER);
        b_color = new GL_Buffer<Vector4>(GL_ARRAY_BUFFER);
        b_normal = new GL_Buffer<Vector3>(GL_ARRAY_BUFFER);
        init_vao();
    }

    void init_vao() {
        vao->bind();
        vao->bind_array(b_vertex, 0, 3, GL_FLOAT);
        vao->bind_array(b_color, 1, 4, GL_FLOAT);
        vao->bind_array(b_normal, 2, 3, GL_FLOAT);
        vao->enable_attribute(0);
        vao->enable_attribute(1);
        vao->enable_attribute(2);
        vao->unbind();
    }

    void clear() {
        geometry.clear();
    }

    void load() {
        b_vertex->load_data(geometry);
    }

    void load_colors()
    {
        assert(vcolors.size() == geometry.size());
        b_color->load_data(vcolors);
    }

    ~GL_Geometry(){

    }
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
    GL_Topology() {
        b_line = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        b_triangle = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        b_quad = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        sbo_tri_to_elem = new GL_Buffer<int>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
        sbo_quad_to_elem = new GL_Buffer<int>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
        sbo_ecolor = new GL_Buffer<Color>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
    }
    
    ~GL_Topology() {
        delete b_line;
        delete b_triangle;
        delete b_quad;
        delete sbo_tri_to_elem;
        delete sbo_quad_to_elem;
    }

    virtual void load()
    {
        if (lines.size() > 0)
            b_line->load_data(lines);
        if (triangles.size() > 0)
            b_triangle->load_data(triangles);
        if (quads.size() > 0)
            b_quad->load_data(quads);
        if (tri_to_elem.size() > 0)
            sbo_tri_to_elem->load_data(tri_to_elem);
        if (quad_to_elem.size() > 0)
            sbo_quad_to_elem->load_data(quad_to_elem);
    }
    
    virtual void laod_ecolor() {
        if (ecolors.size() > 0)
            sbo_ecolor->load_data(ecolors);
    }

    void clear() {
        lines.clear();
        triangles.clear();
        quads.clear();
        tri_to_elem.clear();
        quad_to_elem.clear();
    }

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
    GL_Graphic(const Color& color = ColorBase::Grey(scalar(0.7)))
        : _color(color), _multi_color(false), _element_color(false), _mesh(nullptr),  _gl_geometry(new GL_Geometry())
    { 
    }

    virtual void init() override
    {
        _mesh = entity()->get_component<Mesh>();
    }

    virtual void late_init() override {
        update();
    }

    virtual void update() override
    { 
        
        if (_mesh->need_update() || _mesh->has_dynamic_topology())
        {
            // get and clear the previous displayed topology
            // if gl topo not found create a new one
            for (auto& it : _mesh->topologies()) {
                Element element = it.first;
                if (_gl_topologies.find(element) == _gl_topologies.end()) {
                    _gl_topologies[element] = new GL_Topology();
                }
                GL_Topology* gl_topo = _gl_topologies[element];
                gl_topo->clear();
            }

            // update displayed topology
            update_gl_topology();

            // load it to buffers
            for (auto& it : _mesh->topologies()) {
                _gl_topologies[it.first]->load();
            }
        }

        if (_mesh->need_update() || _mesh->has_dynamic_geometry())
        {
            _gl_geometry->clear();
            update_gl_geometry();
            _gl_geometry->load();
        }

        if (_mesh->need_update() || _mesh->has_dynamic_geometry() || _mesh->has_dynamic_topology()) {
            // if the number of geometry change, we need to change the colors array buffer
            if (_multi_color && !_element_color) {
                update_gl_vcolors(); 
                _gl_geometry->load_colors();
            } // else element color is directly handled by topology update, no change is needded
            else if (_multi_color && _element_color) {
                for (auto& it : _mesh->topologies()) {
                    _gl_topologies[it.first]->laod_ecolor();
                }
            }
        }

        _mesh->need_update() = false;
    }

    void set_ecolors(Element type, std::vector<Color> colors) {
        _gl_topologies[type]->ecolors = colors;
    }

    std::vector<Color>& vcolors() { return _vcolors; }

    void set_vcolors(std::vector<Color> colors) {
        _vcolors = colors;
    }

    Color& color() { return _color; }

    bool use_multi_color() const { return _multi_color; }
    void set_multi_color(bool state) { _multi_color = state; }

    bool use_element_color() const { return _element_color; }
    void set_element_color(bool state) { 
        _element_color = state; 
    }

    std::map<Element, GL_Topology*> gl_topologies() {return _gl_topologies; };
    GL_Geometry* gl_geometry() { return _gl_geometry; };

    virtual ~GL_Graphic() { 
        delete _gl_geometry;
        for (auto& it : _gl_topologies) {
            delete it.second;
        }
        _gl_topologies.clear();
    }

    static scalar wireframe_intencity;
    static Color vertice_color;

protected:

    virtual void update_gl_geometry() {
        _gl_geometry->geometry = _mesh->geometry();
    }

    virtual void update_gl_topology() {
        for (auto& it : _mesh->topologies()) {
            Element element = it.first;
            if (element != Line && element != Triangle && element != Quad) continue;
            
            GL_Topology* gl_topo = _gl_topologies[element];
            switch (element)
            {
                case Line: gl_topo->lines = it.second; break; 
                case Triangle: gl_topo->triangles = it.second; break;
                case Quad: gl_topo->quads = it.second; break;
                default: break;
            }
            if (element == Line) continue;

            Mesh::Topology& face_to_elem = (element == Triangle) ? gl_topo->tri_to_elem : gl_topo->quad_to_elem;
            int nb = elem_nb_vertices(element);
            int nb_element = it.second.size() /nb;
            face_to_elem.resize(nb_element);
            std::iota(std::begin(face_to_elem), std::end(face_to_elem), 0); // (0,1,2,3, ..., n-1)
        }
    }

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

scalar GL_Graphic::wireframe_intencity = scalar(0.7);
Color GL_Graphic::vertice_color = ColorBase::Grey(scalar(0.1));