#pragma once
#include "Core/Component.h"
#include "UI/AppInfo.h"
#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_Program.h"
#include "Rendering/Camera.h"
#include "Manager/TimeManager.h"
#include "Rendering/Renderer.h"
class GL_DisplayMode : public Renderer
{
public:
    GL_DisplayMode() : _graphic(nullptr) { }
    virtual void init() override
    { 
        _graphic     = this->_entity->get_component<GL_Graphic>();
        set_shaders_path(_paths);
        for (int i = 0; i < _paths.size(); ++i)
        {
            _paths[i] = AppInfo::PathToAssets()+ _paths[i];
            _programs.push_back(new GL_Program(_paths[i].c_str()));
        }
    }
    std::vector<std::string> shader_path() { return _paths; }
    virtual ~GL_DisplayMode() { 
        for (GL_Program* prog : _programs) delete prog;   
        _programs.clear();
    }
protected:
    virtual void set_shaders_path(std::vector<std::string>& paths) = 0;

    GL_Graphic* _graphic;
    std::vector<std::string> _paths;
    std::vector<GL_Program*> _programs;
};

class GL_DisplayMesh : public GL_DisplayMode
{
public:
    GL_DisplayMesh() : _wireframe(true), _surface(true), _point(true), _v(Matrix::Identity3x3()), _p(Matrix::Identity3x3()),
    _pos(Unit3D::Zero()) { }

    virtual void update() override {
    } // called before drawing anything

    virtual void draw() override
    {
        GL_Geometry* gl_geometry = _graphic->gl_geometry();
        for (auto& it : _graphic->gl_topologies()) {
            GL_Topology* gl_topo = it.second;

            GL_VAO* vao = gl_geometry->vao;
            GL_Buffer<Vector3>* b_vertices = gl_geometry->b_vertex;
            GL_Buffer<int>* b_line = gl_topo->b_line;
            GL_Buffer<int>* b_triangle = gl_topo->b_triangle;
            GL_Buffer<int>* b_quad = gl_topo->b_quad;

            if (b_vertices->nb_element() <= 0) return;

            _v = Camera::Instance().view();
            _p = Camera::Instance().projection();
            _pos = Camera::Instance().position();

            vao->bind();
            if (_point) draw_vertices(b_vertices);

            if (b_line->nb_element() > 0 && _wireframe)
            {
                draw_line(b_line);
            }

            if (b_triangle->nb_element() > 0)
            {
                if (_wireframe) draw_triangles_wireframe(b_triangle);
                if (_surface) {
                    if (_graphic->use_element_color()) {
                        GL_Buffer<int>* b_tri_to_elem = gl_topo->sbo_tri_to_elem;
                        GL_Buffer<Color>* b_ecolors = gl_topo->sbo_ecolor;
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_tri_to_elem->gl_id());
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_ecolors->gl_id());
                    }
                    draw_triangles(b_triangle);
                }
            }

            if (b_quad->nb_element() > 0) {
                if (_surface) {
                    if (_graphic->use_element_color()) {
                        GL_Buffer<int>* b_quad_to_elem = gl_topo->sbo_quad_to_elem;
                        GL_Buffer<Color>* b_ecolors = gl_topo->sbo_ecolor;
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_quad_to_elem->gl_id());
                        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_ecolors->gl_id());
                    }
                    draw_triangles(b_quad);
                }
            }
            vao->unbind();
        }
        
    }
    
    bool& wireframe() { return _wireframe; }
    bool& surface() { return _surface; }
    bool& point() { return _point; }

protected:
    virtual void set_shaders_path(std::vector<std::string>& paths) override
    {
        // emit (no shading)
        paths.push_back("shaders/emit.glsl");           // 0
        paths.push_back("shaders/emit_vcolors.glsl");   // 1
        
        // flat shading
        paths.push_back("shaders/flat.glsl");           // 2
        paths.push_back("shaders/flat_vcolors.glsl");   // 3
        paths.push_back("shaders/flat_ecolors.glsl");   // 3
    }

    virtual void draw_vertices(GL_Buffer<Vector3>* b_vertices)
    {
        // emit uniform color or use color array buffer is multi color
        int shader_id = _graphic->use_multi_color() && !_graphic->use_element_color() && !_surface && !_wireframe;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        
        if (shader_id == 0)
            this->_programs[shader_id]->uniform("color", GL_Graphic::vertice_color);
 
        glPointSize(5.f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        glDrawArrays(GL_POINTS, 0, b_vertices->nb_element());
        glPointSize(1.f);
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_line(GL_Buffer<int>* b_line)
    { 
        // emit uniform color or use color array buffer is multi color
        int shader_id = _graphic->use_multi_color() && !_graphic->use_element_color();
        _programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 0) {
            _programs[shader_id]->uniform("color", _graphic->color() * GL_Graphic::wireframe_intencity);
        }
        else if(_surface && shader_id == 1) {
            _programs[shader_id]->uniform("wireframe_intencity", GL_Graphic::wireframe_intencity);
        }
        else if (!_surface) {
            _programs[shader_id]->uniform("wireframe_intencity", 1.f);
        }
        b_line->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(2.f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_LINES, b_line->nb_element(), GL_UNSIGNED_INT, 0);
        glDisable(GL_LINE_SMOOTH);
        glLineWidth(1.f);
        b_line->unbind();
        _programs[shader_id]->unbind();
    }

    virtual void draw_triangles(GL_Buffer<int>* b_triangle) {
        int shader_id = 2;
        // use multi color or not
        shader_id += _graphic->use_multi_color();
        shader_id += _graphic->use_element_color();

        _programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 3) {
            _programs[shader_id]->uniform("color", _graphic->color());
        }
        
        b_triangle->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
        glEnable(GL_LINE_SMOOTH);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, b_triangle->nb_element(), GL_UNSIGNED_INT, 0);
        glDisable(GL_LINE_SMOOTH);
        b_triangle->unbind();
        _programs[shader_id]->unbind();
    }

    virtual void draw_triangles_wireframe(GL_Buffer<int>* b_triangle)
    {
        int shader_id = 0;
        shader_id += _graphic->use_multi_color() && !_graphic->use_element_color();

        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 0)
            this->_programs[shader_id]->uniform("color", this->_graphic->color() * GL_Graphic::wireframe_intencity);
        else if (_surface && this->_graphic->use_multi_color()) {
            this->_programs[shader_id]->uniform("wireframe_intencity", GL_Graphic::wireframe_intencity);
        }
        else if (!_surface && this->_graphic->use_multi_color()) {
            this->_programs[shader_id]->uniform("wireframe_intencity", 1.f);
        }
        b_triangle->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);            
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(2.f);
        glEnable(GL_LINE_SMOOTH);
        glDrawElements(GL_TRIANGLES, b_triangle->nb_element(), GL_UNSIGNED_INT, 0);
        glDisable(GL_LINE_SMOOTH);
        glLineWidth(1.f);
        b_triangle->unbind();
        this->_programs[shader_id]->unbind();
    }


    bool _wireframe, _surface, _point;
    Matrix4x4 _v, _p;
    Vector3 _pos;
};


