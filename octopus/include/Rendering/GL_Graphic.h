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
        _sbo_tri_to_elem = new GL_Buffer<int>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
        _sbo_quad_to_elem = new GL_Buffer<int>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
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
        
    }

    virtual void update() override
    { 
        
        if (_mesh->need_update() || _mesh->has_dynamic_topology())
        {
            lines.clear();
            triangles.clear();
            quads.clear();
            get_topology(lines, triangles, quads, tri_to_elem, quad_to_elem);
            update_buffer_topology(lines, triangles, quads, tri_to_elem, quad_to_elem);
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
    GL_Buffer<int>* b_tri_to_elem() { return _sbo_tri_to_elem; }
    GL_Buffer<int>* b_quad_to_elem() { return _sbo_quad_to_elem; }

    Mesh::Topology& get_lines() { return lines; }
    Mesh::Topology& get_triangles() { return triangles; }
    Mesh::Topology& get_quads() { return quads; }
    Mesh::Geometry& get_geometry() { return geometry; }

    bool use_multi_color() { return _multi_color; }
    void set_multi_color(bool state) { _multi_color = state; }

    bool use_element_color() { return _element_color; }
    void set_element_color(bool state) { _element_color = state; }

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

    virtual void get_topology(
        Mesh::Topology& lines, 
        Mesh::Topology& triangles, 
        Mesh::Topology& quads, 
        Mesh::Topology& tri_to_elem, 
        Mesh::Topology& quad_to_elem) {
        lines     = _mesh->topology(Line);
        triangles = _mesh->topology(Triangle);

        Mesh::Topology& element_quads = _mesh->topology(Quad);

        quads.resize(element_quads.size() / 4 * 6);
        quad_to_elem.resize(quads.size() / 3);

        int quad_lines[8] = { 0,1,1,2,2,3,3,0 };
        int quad_triangle[6] = { 0,1,3, 3,1,2 };
        for (int i = 0; i < element_quads.size() / 4; i++)
        {
            for (int j = 0; j < 8; ++j)
                lines[i * 8 + j] = element_quads[i * 4 + quad_lines[j]];

            for (int j = 0; j < 6; ++j) {
                quads[i * 6 + j] = element_quads[i * 4 + quad_triangle[j]];
            }
            quad_to_elem[i*2] = i;
            quad_to_elem[i*2+1] = i;
        }

        tri_to_elem.resize(triangles.size()/3);
        for (int i = 0; i < tri_to_elem.size(); i++) {
            tri_to_elem[i] = i;
        }

        quad_to_elem.resize(quads.size()/3);
        for (int i = 0; i < quad_to_elem.size(); i++) {
            quad_to_elem[i] = i;
        }
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
        Vector3 v[3]{};
        int vid[3]{};
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

    virtual void update_buffer_topology(
        const Mesh::Topology& lines, 
        const Mesh::Topology& triangles, 
        const Mesh::Topology& quads, 
        const Mesh::Topology& tri_to_elem, 
        const Mesh::Topology& quad_to_elem)
    {
        if (lines.size() > 0)
            _b_line->load_data(lines);
        if (triangles.size() > 0)
            _b_triangle->load_data(triangles);
        if (quads.size() > 0)
            _b_quad->load_data(quads);
        if (tri_to_elem.size() > 0)
            _sbo_tri_to_elem->load_data(tri_to_elem);
        if (quad_to_elem.size() > 0)
            _sbo_quad_to_elem->load_data(quad_to_elem);
    }

    virtual void update_buffer_colors()
    {
        if (_mesh->nb_vertices() == 0) return;
        if (!_element_color) {
            assert(_colors.size() == _mesh->nb_vertices());
        }
        else {
            // check if colors.size == nb_element
        }
        _b_color->load_data(_colors);
    }

    virtual void update_buffer_normals(const std::vector<Vector3>& v_normals) 
    {
        if (v_normals.size() == 0) return;
        _b_normal->load_data(v_normals);
    }

    bool _multi_color;
    bool _element_color;
    bool _normals;
    Color _color;
    std::vector<Color> _colors;

    Mesh* _mesh;
    GL_VAO* _vao;
    GL_Buffer<Vector3>* _b_normal;
    GL_Buffer<Vector3>* _b_vertex;
    GL_Buffer<Color>* _b_color;
    GL_Buffer<int> *_b_line, *_b_triangle, *_b_quad;
    GL_Buffer<int> *_sbo_tri_to_elem, * _sbo_quad_to_elem;
private:
    Mesh::Topology lines, triangles, quads;
    Mesh::Topology tri_to_elem, quad_to_elem;
    Mesh::Geometry geometry;
};

scalar GL_Graphic::wireframe_intencity = scalar(0.7);
Color GL_Graphic::vertice_color = ColorBase::Grey(scalar(0.1));