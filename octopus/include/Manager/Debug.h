#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Mesh/Mesh.h"
#include <vector>
class Debug : public Singleton<Debug>
{
protected:
    friend Singleton<Debug>;
    Debug() : _current_color(ColorBase::White()), _mesh(nullptr), _graphic(nullptr) { }

    void set_current_color(const Color& color) {
        _current_color = color;
    }

    void add_vertice(const Vector3& p) { 
        assert(_mesh != nullptr);
        assert(_graphic != nullptr);
        _mesh->geometry().push_back(p);
        _graphic->colors().push_back(_current_color);
    }

    void add_line(unsigned int a, unsigned int b) { 
        assert(_mesh != nullptr);
        _mesh->topology(Element::Line).push_back(a);
        _mesh->topology(Element::Line).push_back(b);
    }

    Color _current_color;
    Mesh* _mesh;
    GL_Graphic* _graphic;

public:
    void set_mesh(Mesh* mesh) { _mesh = mesh; }
    void set_graphic(GL_Graphic* graphic) { _graphic = graphic; }
    void clear()
    {
        _mesh->clear();
        _graphic->colors().clear();
    }

    static void SetColor(const Color& color);
    static void Axis(const Vector3& p, scalar length);
    static void Axis(const Vector3& p, const Matrix4x4 rot, scalar length);
    static void Line(const Vector3& a, const Vector3& b);
    static void Vector(const Vector3& p, const Vector3& direction, scalar length);
    static void UnitGrid(unsigned int _size);
    static void Cube(const Vector3& p_min, const Vector3 p_max);
    static void Cube(const Vector3& p, scalar size);
};

void Debug::SetColor(const Color& color)
{
    Instance().set_current_color(color);
}

void Debug::Line(const Vector3& a, const Vector3& b) {
    Debug* debug = Instance_ptr();
    unsigned int _i_start = debug->_mesh->nb_vertices();
    debug->add_vertice(a);
    debug->add_vertice(b);
    debug->add_line(_i_start, _i_start + 1);
}

void Debug::Vector(const Vector3& p, const Vector3& direction, scalar length = 1.)
{
    Line(p, p + direction * length);
}

void Debug::UnitGrid(unsigned int _size)
{
    _size = std::max(0u, _size);
    unsigned int nb_square = (_size + 1u) * 2u;
    Vector3 o(-scalar(_size + 1u), 0., -scalar(_size + 1u));
    Vector3 dir_x = Unit3D::right() * scalar(nb_square);
    Vector3 dir_z = Unit3D::forward() * scalar(nb_square);
    for (unsigned int i = 0; i <= nb_square; ++i) {
        Vector(o + Unit3D::right() * scalar(i), dir_z);
        Vector(o + Unit3D::forward() * scalar(i), dir_x);
    }
}

void Debug::Axis(const Vector3& p, const Matrix4x4 rot, scalar length = 1.) {
    Matrix3x3 r = rot;
    Color color = Instance()._current_color;
    SetColor(ColorBase::Red());
    Vector(p, r * Unit3D::right() * length);
    SetColor(ColorBase::Green());
    Vector(p, r * Unit3D::up() * length);
    SetColor(ColorBase::Blue());
    Vector(p, r * Unit3D::forward() * length);
    SetColor(color);
}

void Debug::Axis(const Vector3& p, scalar length = 1.) { 
    Debug::Axis(p, Matrix::Identity4x4(), length);
}

void Debug::Cube(const Vector3& p_min, const Vector3 p_max)
{
    Debug* debug = Instance_ptr();
    unsigned int _i_start = debug->_mesh->nb_vertices();
    for (int i = 0; i < 8; ++i)
    {
        Vector3 v(p_min);
        if (i & 1) v.x = p_max.x;
        if (i & 2) v.y = p_max.y;
        if (i & 4) v.z = p_max.z;
        debug->add_vertice(v);
    }
    static unsigned int box_topo[24] = { 0, 1, 1, 3, 3, 2, 2, 0, 2, 6, 6, 7,
                                    7, 3, 7, 5, 5, 1, 6, 4, 4, 0, 4, 5 };
    for (unsigned int i = 0; i < 24; i += 2)
    {
        debug->add_line(_i_start + box_topo[i], _i_start + box_topo[i + 1]);
    }

}


void Debug::Cube(const Vector3& p = Unit3D::Zero(), scalar size = scalar(1.f))
{
    Vector3 p_min(p - Vector3(size) * scalar(0.5));
    Vector3 p_max(p + Vector3(size) * scalar(0.5));
    Debug::Cube(p_min, p_max);
}

