#pragma once
#include "Core/Component.h"
#include "HUD/AppInfo.h"
#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_Program.h"
#include "Rendering/Camera.h"
#include "Manager/TimeManager.h"
class GL_DisplayMode : public Component
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
    virtual void draw() = 0;
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
    GL_DisplayMesh() : _wireframe(true), _surface(true), _point(true) { 
    }

    virtual void update() override { } // called before drawing anything

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

        if (b_line->nb_element() > 0 && _wireframe)
        {
            draw_line(b_line);
        }

        if (b_triangle->nb_element() > 0)
        {
            if (_wireframe) draw_triangles_wireframe(b_triangle);
            if (_surface) draw_triangles(b_triangle);
        }

        if (b_quad->nb_element() > 0 && _surface)
        {
            draw_triangles(b_quad);
        }

        vao->unbind();
    }

    void draw_wireframe(bool state) { _wireframe = state; }
    void draw_surface(bool state) { _surface = state; }
    void draw_points(bool state) { _point = state; }

protected:
    virtual void set_shaders_path(std::vector<std::string>& paths) override
    {
        paths.push_back("shaders/mesh.glsl");
        paths.push_back("shaders/mesh_colors.glsl");
    }

    virtual void draw_vertices(GL_Buffer<Vector3>* b_vertices)
    {
        unsigned int shader_id = this->_graphic->use_multi_color() && !_surface && !_wireframe;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        
        if (shader_id == 0)
            this->_programs[0]->uniform("mesh_color", ColorBase::Grey(0.2));
 
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
            this->_programs[0]->uniform("mesh_color", ColorBase::Grey(0.8));
        b_line->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
        glEnable(GL_LINE_SMOOTH);
        glLineWidth(3.f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawElements(GL_LINES, b_line->nb_element(), GL_UNSIGNED_INT, 0);
        glDisable(GL_LINE_SMOOTH);
        glLineWidth(1.f);
        b_line->unbind();
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_triangles(GL_Buffer<unsigned int>* b_triangle) {
        unsigned int shader_id = this->_graphic->use_multi_color();
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 0)
            this->_programs[0]->uniform("mesh_color", _graphic->color());

        b_triangle->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
        glEnable(GL_LINE_SMOOTH);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, b_triangle->nb_element(), GL_UNSIGNED_INT, 0);
        b_triangle->unbind();
        this->_programs[shader_id]->unbind();
    }

    virtual void draw_triangles_wireframe(GL_Buffer<unsigned int>* b_triangle)
    {
        unsigned int shader_id = this->_graphic->use_multi_color() && !_surface;
        this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
        if (shader_id == 0)
            this->_programs[0]->uniform("mesh_color", ColorBase::Grey(0.8));

        b_triangle->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);            
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(3.f);
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