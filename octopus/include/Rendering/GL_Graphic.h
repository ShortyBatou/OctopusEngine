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
    GL_Graphic(const Color& color = ColorBase::Grey(scalar(0.7)))
        : _color(color), _multi_color(false), _mesh(nullptr)
    { 
        init_buffers();
        init_vao();
    }

    virtual void init() override
    {
        Entity* e = this->entity();
        _mesh     = e->get_component<Mesh>();
        
    }

    void init_buffers() {
        _vao        = new GL_VAO();
        _b_vertex   = new GL_Buffer<Vector3>(GL_ARRAY_BUFFER);
        _b_color    = new GL_Buffer<Vector4>(GL_ARRAY_BUFFER);
        _b_normal   = new GL_Buffer<Vector3>(GL_ARRAY_BUFFER);
        _b_line     = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        _b_triangle = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
        _b_quad     = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    }

    void init_vao() { 
        _vao->bind();
        _vao->bind_array(_b_vertex, 0, 3, GL_FLOAT);
        _vao->bind_array(_b_color,  1, 4, GL_FLOAT);
        _vao->bind_array(_b_normal, 2, 3, GL_FLOAT);
        _vao->enable_attribute(0);
        _vao->enable_attribute(1);
        _vao->enable_attribute(2);
        _vao->unbind();
    }

    virtual void late_init() override {
        update();
    }

    virtual void update() override
    { 
        

        if (_mesh->need_update() || _mesh->has_dynamic_topology())
        {
            lines.clear();
            triangles.clear();
            quads.clear();
            get_topology(lines, triangles, quads);
            update_buffer_topology(lines, triangles, quads);
        }

        if (_mesh->need_update() || _mesh->has_dynamic_geometry())
        {
            geometry.clear();
            get_geometry(geometry);
            update_buffer_geometry(geometry);
            if (_multi_color) update_buffer_colors();
        }

        if (_normals) {
            std::vector<Vector3> v_normals;
            get_normals(geometry, triangles, quads, v_normals);
            update_buffer_normals(v_normals);
        }

        _mesh->need_update() = false;
    }

    Color& color() { return _color; }
    std::vector<Color>& colors() { return _colors; }
    GL_VAO* vao() { return _vao; }
    GL_Buffer<Vector3>* b_vertex() { return _b_vertex; }
    GL_Buffer<Vector3>* b_normal() { return _b_normal; }
    GL_Buffer<Vector4>* b_color() { return _b_color; }
    GL_Buffer<int>* b_line() { return _b_line; }
    GL_Buffer<int>* b_triangle() { return _b_triangle; }
    GL_Buffer<int>* b_quad() { return _b_quad; }

    Mesh::Topology& get_lines() { return lines; }
    Mesh::Topology& get_triangles() { return triangles; }
    Mesh::Topology& get_quads() { return quads; }
    Mesh::Geometry& get_geometry() { return geometry; }

    bool use_multi_color() { return _multi_color; }
    void set_multi_color(bool state)
    {
        _multi_color = state;
    }
    bool& normals() { return _normals; }

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

    virtual void get_geometry(Mesh::Geometry& geometry) {
        geometry = _mesh->geometry();
    }

    virtual void get_topology(Mesh::Topology& lines, Mesh::Topology& triangles, Mesh::Topology& quads) {
        lines     = _mesh->topology(Line);
        triangles = _mesh->topology(Triangle);
        quads     = _mesh->topology(Quad);
    }

    virtual void get_normals(const Mesh::Geometry& geometry, const Mesh::Topology& triangles, const Mesh::Topology& quads, std::vector<Vector3>& v_normals)
    {
        v_normals.resize(geometry.size(), Unit3D::Zero());
        
        for (int i = 0; i < triangles.size(); i += 3) {
            compute_vertex_normals(geometry, triangles, i, v_normals);
        }

        for (int i = 0; i < quads.size(); i += 3) {
            compute_vertex_normals(geometry, quads, i, v_normals);
        }

        for (int i = 0; i < v_normals.size(); ++i) {
            v_normals[i] = glm::normalize(v_normals[i]);
        }

    }
    virtual void compute_vertex_normals(const Mesh::Geometry& geometry, const Mesh::Topology& triangles, int i, std::vector<Vector3>& v_normals) {
        Vector3 v[3];
        int vid[3];
        for (int j = 0; j < 3; ++j) {
            vid[j] = triangles[i + j];
            v[j] = geometry[vid[j]];
        }

        Vector3 t_normal = glm::cross(v[1] - v[0], v[2] - v[0]) * scalar(0.5);
        for (int j = 0; j < 3; ++j) {
            v_normals[vid[j]] += t_normal;
        }
    }

    virtual void update_buffer_geometry(const Mesh::Geometry& geometry)
    {
        _b_vertex->load_data(geometry);
    }

    virtual void update_buffer_topology(const Mesh::Topology& lines, const Mesh::Topology& triangles, const Mesh::Topology& quads)
    {
        if (lines.size() > 0)
            _b_line->load_data(lines);
        if (triangles.size() > 0)
            _b_triangle->load_data(triangles);
        if (quads.size() > 0)
            _b_quad->load_data(quads);
    }

    virtual void update_buffer_colors()
    {
        assert(_colors.size() == _mesh->nb_vertices());
        if (_mesh->nb_vertices() == 0) return;
        _b_color->load_data(_colors);
    }

    virtual void update_buffer_normals(const std::vector<Vector3>& v_normals) 
    {
        if (v_normals.size() == 0) return;
        _b_normal->load_data(v_normals);
    }

    bool _multi_color;
    bool _normals;
    Color _color;
    std::vector<Color> _colors;

    Mesh* _mesh;
    GL_VAO* _vao;
    GL_Buffer<Vector3>* _b_normal;
    GL_Buffer<Vector3>* _b_vertex;
    GL_Buffer<Color>* _b_color;
    GL_Buffer<int> *_b_line, *_b_triangle, *_b_quad;

private:
    Mesh::Topology lines, triangles, quads;
    Mesh::Geometry geometry;
};

scalar GL_Graphic::wireframe_intencity = scalar(0.7);
Color GL_Graphic::vertice_color = ColorBase::Grey(scalar(0.1));