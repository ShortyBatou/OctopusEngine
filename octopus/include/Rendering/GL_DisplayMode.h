#pragma once
#include "Rendering/Camera.h"
#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_Program.h"
#include "Manager/TimeManager.h"
#include "Rendering/Renderer.h"

class GL_DisplayMode : public Renderer {
public:
    GL_DisplayMode() : _graphic(nullptr) {
    }

    void init() override;

    std::vector<std::string> shader_path() { return _paths; }
    void set_graphic(GL_Graphic *graphic) { _graphic = graphic; }

    ~GL_DisplayMode() override {
        for (GL_Program *prog: _programs) delete prog;
        _programs.clear();
    }

protected:
    virtual void set_shaders_path(std::vector<std::string> &paths) = 0;

    GL_Graphic *_graphic;
    std::vector<std::string> _paths;
    std::vector<GL_Program *> _programs;
};

class GL_DisplayMesh final : public GL_DisplayMode {
public:
    GL_DisplayMesh() : _wireframe(true), _surface(true), _point(true), _v(Matrix::Identity3x3()),
                       _p(Matrix::Identity3x3()), _pos(Unit3D::Zero()) {
    }


    void draw() override;

    bool &wireframe() { return _wireframe; }
    bool &surface() { return _surface; }
    bool &point() { return _point; }

protected:
    void set_shaders_path(std::vector<std::string> &paths) override;

    void draw_vertices(const GL_Buffer<Vector3> *b_vertices) const;

    void draw_line(GL_Buffer<int> *b_line) const;

    void draw_triangles(GL_Buffer<int> *b_triangle) const;

    void draw_triangles_wireframe(GL_Buffer<int> *b_triangle) const;

    bool _wireframe, _surface, _point;
    Matrix4x4 _v, _p;
    Vector3 _pos;
};
