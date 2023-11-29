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
        _graphic     = this->_entity->getComponent<GL_Graphic>();
        set_shaders_path(_paths);
        for (unsigned int i = 0; i < _paths.size(); ++i)
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
    GL_DisplayMesh() 
        : _wireframe(true), _surface(true), _point(true), _normal(false), 
          _normal_color(ColorBase::Blue()), _normal_length(0.05) {
    }

    virtual void update() override {
    } // called before drawing anything

    virtual void draw() override
    {
        GL_VAO* vao = this->_graphic->vao();

        GL_Buffer<Vector3>* b_vertices      = this->_graphic->b_vertex();
        GL_Buffer<unsigned int>* b_line     = this->_graphic->b_line();
        GL_Buffer<unsigned int>* b_triangle = this->_graphic->b_triangle();
        GL_Buffer<unsigned int>* b_quad     = this->_graphic->b_quad();

        if (b_vertices->nb_element() <= 0) return;

        _v = Camera::Instance().view();
        _p = Camera::Instance().projection();
        _pos = Camera::Instance().position();
        
        vao->bind();
        if(_point) draw_vertices(b_vertices);

        if (_normal && this->_graphic->normals())
            draw_vertices_normals(b_vertices);

        if (b_line->nb_element() > 0 && _wireframe)
        {
            draw_line(b_line);
        }

        if (b_triangle->nb_element() > 0)
        {
            if (_wireframe) draw_triangles_wireframe(b_triangle);
            if (_surface) draw_triangles(b_triangle);
            if (_normal && !this->_graphic->normals()) draw_face_normals(b_triangle);
        }

        if (b_quad->nb_element() > 0 && _surface)
        {
            draw_triangles(b_quad);
            if (_normal && !this->_graphic->normals()) draw_face_normals(b_quad);
        }

        vao->unbind();
    }
    
    bool& wireframe() { return _wireframe; }
    bool& surface() { return _surface; }
    bool& point() { return _point; }
    bool& normal() { return _normal; }
    Color& normal_color() { return _normal_color; }
    scalar& normal_length() { return _normal_length; }

protected:
    virtual void set_shaders_path(std::vector<std::string>& paths) override
    {
        // emit (no shading)
        paths.push_back("shaders/mesh_emit.glsl");
        paths.push_back("shaders/mesh_colors_emit.glsl");
        
        // flat shading
        paths.push_back("shaders/mesh_flat.glsl");
        paths.push_back("shaders/mesh_colors_flat.glsl");
        paths.push_back("shaders/mesh_flat_normal.glsl");
        
        // smooth shading
        paths.push_back("shaders/mesh_smooth.glsl");
        paths.push_back("shaders/mesh_colors_smooth.glsl");
        paths.push_back("shaders/mesh_normal.glsl");
    }

    virtual void draw_face_normals(GL_Buffer<unsigned int>* b_triangles) 
    {
        unsigned int shader_id = 4;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        this->_programs[shader_id]->uniform("color", _normal_color);
        this->_programs[shader_id]->uniform("n_length", _normal_length);
        b_triangles->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(2.f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, b_triangles->nb_element(), GL_UNSIGNED_INT, 0);
        glLineWidth(1.f);
        glDisable(GL_LINE_SMOOTH);
        b_triangles->unbind();
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_vertices_normals(GL_Buffer<Vector3>* b_vertices)
    {
        unsigned int shader_id = 7;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());

        this->_programs[shader_id]->uniform("color", _normal_color);
        this->_programs[shader_id]->uniform("n_length", _normal_length);

        glLineWidth(2.f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        glDrawArrays(GL_POINTS, 0, b_vertices->nb_element());
        glLineWidth(1.f);
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_vertices(GL_Buffer<Vector3>* b_vertices)
    {
        unsigned int shader_id = this->_graphic->use_multi_color() && !_surface && !_wireframe;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        
        if (shader_id == 0)
            this->_programs[shader_id]->uniform("color", GL_Graphic::vertice_color);
 
        glPointSize(5.f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        glDrawArrays(GL_POINTS, 0, b_vertices->nb_element());
        glPointSize(1.f);
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_line(GL_Buffer<unsigned int>* b_line)
    { 
        unsigned int shader_id = this->_graphic->use_multi_color() && !_surface;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 0)
            this->_programs[shader_id]->uniform("color", this->_graphic->color() * GL_Graphic::wireframe_intencity);

        b_line->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(2.f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_LINES, b_line->nb_element(), GL_UNSIGNED_INT, 0);
        glDisable(GL_LINE_SMOOTH);
        glLineWidth(1.f);
        b_line->unbind();
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_triangles(GL_Buffer<unsigned int>* b_triangle) {
        unsigned int shader_id = this->_graphic->use_multi_color();
        shader_id += this->_graphic->normals() ? 5 : 2;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 2 || shader_id == 5)
            this->_programs[shader_id]->uniform("color", _graphic->color());

        b_triangle->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
        glEnable(GL_LINE_SMOOTH);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, b_triangle->nb_element(), GL_UNSIGNED_INT, 0);
        glDisable(GL_LINE_SMOOTH);
        b_triangle->unbind();
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_triangles_wireframe(GL_Buffer<unsigned int>* b_triangle)
    {
        unsigned int shader_id = this->_graphic->use_multi_color() && !_surface;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 0)
            this->_programs[0]->uniform("color", this->_graphic->color() * GL_Graphic::wireframe_intencity);

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


    bool _wireframe, _surface, _point, _normal;
    Color _normal_color;
    scalar _normal_length;
    Matrix4x4 _v, _p;
    Vector3 _pos;
};


