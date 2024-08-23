#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Mesh/Mesh.h"
#include "Tools/Color.h"
#include "Rendering/GL_Graphic.h"
#include <vector>
#include <map>
#include <iomanip>


class Debug : public Singleton<Debug> {
protected:
    friend Singleton;

    Debug() : _current_color(ColorBase::White()), _mesh(nullptr), _graphic(nullptr) {
    }

    void set_current_color(const Color &color) {
        _current_color = color;
    }

    void add_vertice(const Vector3 &p);

    void add_line(int a, int b);

    std::vector<Color> vcolors;
    Color _current_color;
    Mesh *_mesh;
    GL_Graphic *_graphic;

public:
    void set_mesh(Mesh *mesh) { _mesh = mesh; }
    void set_graphic(GL_Graphic *graphic) { _graphic = graphic; }

    void clear() const;

    static void SetColor(const Color &color);

    static void Axis(const Vector3 &p, scalar length);

    static void Axis(const Vector3 &p, const Matrix4x4 &rot, scalar length);

    static void Line(const Vector3 &a, const Vector3 &b);

    static void Vector(const Vector3 &p, const Vector3 &direction, scalar length);

    static void UnitGrid(int _size);

    static void Cube(const Vector3 &p_min, const Vector3 &p_max);

    static void Cube(const Vector3 &p, scalar size);
};


struct DebugUI_Component {
    explicit DebugUI_Component(const std::string &_name) : name(_name) {
    }

    virtual void draw(int w, int h) = 0;

    virtual ~DebugUI_Component() = default;

    std::string name;

    /// MUST BE ELSEWHERE !
    static std::string convert_scientific(double value, int precision = 3);
};


struct DebugUI_Group final : DebugUI_Component {
    explicit DebugUI_Group(const std::string &_name) : DebugUI_Component(_name) {
    }

    void draw(int w, int h) override;

    ~DebugUI_Group() override;

    std::map<std::string, DebugUI_Component *> ui_components;
};


struct DebugUI_Plot final : DebugUI_Component {
    explicit DebugUI_Plot(const std::string &name, int size, bool auto_range = true)
        : DebugUI_Component(name), _auto_range(auto_range), _r_min(std::numeric_limits<float>::max()),
          _r_max(std::numeric_limits<float>::min()),
          _vmin(0), _vmax(0), _offset(0), _size(size) {
        _values.resize(_size);
    }

    void draw(int w, int h) override;

    void add_value(float value);

    void set_range(float r_min, float r_max) {
        _r_min = r_min;
        _r_max = r_max;
    }

protected:
    bool _auto_range;
    float _r_min, _r_max;
    float _vmin, _vmax;
    int _offset;
    int _size;
    std::vector<float> _values;
};


struct DebugUI_Range final : DebugUI_Component {
    explicit DebugUI_Range(const std::string &name)
        : DebugUI_Component(name),
          _vmin(std::numeric_limits<float>::max()), _vmax(std::numeric_limits<float>::min()), _vmean(0),
          _vsum(0), _nb_values(0) {
    }


    void draw(int w, int h) override;

    void add_value(float value);

protected:
    float _vmin, _vmax, _vmean, _vsum;
    int _nb_values;
};


struct DebugUI_Value final : DebugUI_Component {
    explicit DebugUI_Value(const std::string &name)
        : DebugUI_Component(name), _value(0) {
    }

    void draw(int w, int h) override;

    void set_value(float value) {
        _value = value;
    }

protected:
    float _value;
};


class DebugUI final : public Singleton<DebugUI> {
    friend Singleton;
    DebugUI() { current_group = ""; }
    std::map<std::string, DebugUI_Group *> ui_groups;
    std::string current_group;

public:
    void draw() const;

    DebugUI_Group *get_group() {
        return ui_groups[current_group];
    }

    virtual ~DebugUI();

    static void Begin(const std::string &name);

    static void Plot(const std::string &name, const float &value, int buffer);

    static void Plot(const std::string &name, const float &value, float r_min, float r_max, int buffer);

    static void Value(const std::string &name, const float &value);

    static void Range(const std::string &name, const float &value);

    static void End();
};
