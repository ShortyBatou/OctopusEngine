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

class GL_Graphic : public Component
{
public:
    GL_Graphic(const Color& color = ColorBase::Grey(0.7))
        : _color(color), _multi_color(false), _mesh(nullptr)
    { 
        init_buffers();
        init_vao();
    }

    virtual void init() override
    {
        Entity* e = this->entity();
        _mesh     = e->getComponent<Mesh>();
        
    }

    void init_buffers() {
        _vao        = new GL_VAO();
        _b_vertex   = new GL_Buffer<Vector3>(GL_ARRAY_BUFFER);
        _b_color    = new GL_Buffer<Vector4>(GL_ARRAY_BUFFER);
        _b_line     = new GL_Buffer<unsigned int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        _b_triangle = new GL_Buffer<unsigned int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        _b_quad     = new GL_Buffer<unsigned int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    }

    void init_vao() { 
        _vao->bind();
        _vao->bind_array(_b_vertex, 0, 3, GL_FLOAT);
        _vao->bind_array(_b_color, 1, 4, GL_FLOAT);
        _vao->enable_attribute(0);
        _vao->enable_attribute(1);
        _vao->unbind();
    }

    virtual void late_init() override {
        update();
    }

    virtual void update() override
    { 
        if (_mesh->need_update() || _mesh->has_dynamic_geometry())
        {
            update_buffer_geometry();
            if (_multi_color) update_buffer_colors();
        }

        if (_mesh->need_update() ||  _mesh->has_dynamic_topology())
        {
            update_buffer_topology();
        }

        _mesh->need_update() = false;
    }

    Color& color() { return _color; }
    std::vector<Color>& colors() { return _colors; }
    GL_VAO* vao() { return _vao; }
    GL_Buffer<Vector3>* b_vertex() { return _b_vertex; }
    GL_Buffer<Vector4>* b_color() { return _b_color; }
    GL_Buffer<unsigned int>* b_line() { return _b_line; }
    GL_Buffer<unsigned int>* b_triangle() { return _b_triangle; }
    GL_Buffer<unsigned int>* b_quad() { return _b_quad; }
    bool use_multi_color() { return _multi_color; }
    void set_multi_color(bool state)
    {
        _multi_color = state;
    }
    virtual ~GL_Graphic() { 
        delete _vao;
        delete _b_vertex;
        delete _b_color;
        delete _b_line;
        delete _b_triangle;
        delete _b_quad;
    }

    static scalar wireframe_intencity;
    static Color vertice_color;

protected:
    virtual void update_buffer_geometry()
    {
        const Mesh::Geometry& geometry = _mesh->geometry();
        _b_vertex->load_data(geometry);
    }

    virtual void update_buffer_topology()
    {
        if (_mesh->topologies()[Line].size() > 0) 
            _b_line->load_data(_mesh->topologies()[Line]);
        if (_mesh->topologies()[Triangle].size() > 0)
            _b_triangle->load_data(_mesh->topologies()[Triangle]);
    }

    virtual void update_buffer_colors()
    {
        assert(_colors.size() == _mesh->nb_vertices());
        if (_mesh->nb_vertices() == 0) return;
        _b_color->load_data(_colors);
    }

    bool _multi_color;
    Color _color;
    std::vector<Color> _colors;

    Mesh* _mesh;
    GL_VAO* _vao;
    GL_Buffer<Vector3>* _b_vertex;
    GL_Buffer<Color>* _b_color;
    GL_Buffer<unsigned int> *_b_line, *_b_triangle, *_b_quad;
};

scalar GL_Graphic::wireframe_intencity = 0.7;
Color GL_Graphic::vertice_color = ColorBase::Grey(0.1);