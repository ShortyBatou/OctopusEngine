#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Mesh/Mesh.h"
#include "imgui.h"
#include <vector>
#include <unordered_map>
#include <map>
#include <iomanip>
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



struct DebugUI_Component {
    DebugUI_Component(std::string _name) : name(_name) {}
    virtual void draw(unsigned int w, unsigned int h) = 0;
    virtual ~DebugUI_Component() {}
    std::string name;

    /// MUST BE ELSEWHERE !
    std::string convert_scientific(double value, unsigned int precision = 1)
    {
        std::ostringstream str;
        str << std::scientific << std::setprecision(precision) << value;
        std::string s = str.str();
        return s;
    }

};


struct DebugUI_Group : public DebugUI_Component {
    DebugUI_Group(std::string _name) : DebugUI_Component(_name) {}
    virtual void draw(unsigned int w, unsigned int h) {
        if (ImGui::TreeNodeEx(this->name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            for (auto ui : ui_components) {
                ui.second->draw(w, h);
            }
            ImGui::TreePop();
            ImGui::Spacing();
        }
        
    }

    virtual ~DebugUI_Group() {
        for (auto ui : ui_components) {
            delete ui.second;
        }
        ui_components.clear();
    }

    std::map<std::string, DebugUI_Component*> ui_components;
};


struct DebugUI_Plot : public DebugUI_Component {
    DebugUI_Plot(std::string name, unsigned int size, bool auto_range = true)
        : DebugUI_Component(name), _size(size), _offset(0), _auto_range(auto_range),
          _r_min(std::numeric_limits<float>::max()), _r_max(std::numeric_limits<float>::min())
    {
        _values.resize(_size);
    }
    
    virtual void draw(unsigned int w, unsigned int h) {
        ImGui::PlotLines("", _values.data(), _size, _offset, "", _r_min, _r_max, ImVec2(w * 0.8, 60));
    }

    void add_value(float value) {
        _values[_offset] = value;
        _offset = (_offset + 1) % _size;
        if (_auto_range) {
            _r_min = std::min(value, _r_min);
            _r_max = std::max(value, _r_max);
        }
    }

    void set_range(float r_min, float r_max) {
        _r_min = r_min;
        _r_max = r_max;
    }

protected:
    bool _auto_range;
    float _r_min, _r_max;
    float _vmin, _vmax;
    unsigned int _offset;
    unsigned int _size;
    std::vector<float> _values;
};


struct DebugUI_Range: public DebugUI_Component {
    DebugUI_Range(std::string name)
        : DebugUI_Component(name), 
        _vmean(0), _vsum(0), _nb_values(0),
        _vmin(std::numeric_limits<float>::max()), _vmax(std::numeric_limits<float>::min())
    { }


   
    virtual void draw(unsigned int w, unsigned int h) {
        std::string s_min = convert_scientific(_vmin);
        std::string s_max = convert_scientific(_vmax);
        std::string s_mean = convert_scientific(_vmean);
        ImGui::Text("Min = %s \t Max = %s \t Mean = %s", s_min.c_str(), s_max.c_str(), s_mean.c_str());
    }

    void add_value(float value) {
        ++_nb_values;
        _vsum += value;
        _vmean = _vsum / _nb_values;
        _vmin = std::min(value, _vmin);
        _vmax = std::max(value, _vmax);
    }

protected:
    float _vmin, _vmax, _vmean, _vsum;
    unsigned int _nb_values;
};


struct DebugUI_Value : public DebugUI_Component {
    DebugUI_Value(std::string name)
        : DebugUI_Component(name), _value(0) { }

    virtual void draw(unsigned int w, unsigned int h) {
        std::string s_val = convert_scientific(_value);
        ImGui::Text("%s = %s", this->name.c_str(), s_val.c_str());
    }

    void set_value(float value) {
        _value = value;
    }

protected:
    float _value;
};


class DebugUI : public Singleton<DebugUI> {
    friend Singleton<DebugUI>;
    DebugUI() { current_group = ""; }
    std::map<std::string, DebugUI_Group*> ui_groups;
    std::string current_group;
public:
    virtual void draw() {

        unsigned int w, h;
        AppInfo::Window_sizes(w, h);
        ImGui::SetNextWindowPos(ImVec2(w-410,10));
        unsigned int sx = 400;
        unsigned int sy = std::max(int(h) - 10, 10);
        ImGui::SetNextWindowSize(ImVec2(sx, sy));
        ImGui::Begin("Debug");
        for (auto ui : ui_groups) {
            ui.second->draw(sx, sy);
        }

        ImGui::End();
    }

    DebugUI_Group* get_group() {
        return ui_groups[current_group];
    }

    virtual ~DebugUI() {
        for (auto c : ui_groups) {
            delete c.second;
        }
        ui_groups.clear();
    }
    static void Begin(const std::string& name);
    static void Plot(const std::string& name, const float& value, int buffer);
    static void Plot(const std::string& name, const float& value, float r_min, float r_max, int buffer);
    static void Value(const std::string& name, const float& value);
    static void Range(const std::string& name, const float& value);
    static void End();
};

void DebugUI::Begin(const std::string& name) 
{
    DebugUI* ui_debug = Instance_ptr();
    if (ui_debug->ui_groups.find(name) == ui_debug->ui_groups.end()) {
        ui_debug->ui_groups[name] = new DebugUI_Group(name);
    }
    Instance_ptr()->current_group = name;
}

void DebugUI::End()
{
    Instance_ptr()->current_group = "";
}

void DebugUI::Plot(const std::string& name, const float& value, int buffer = 60)
{
    DebugUI::Plot(name, value, 0, 0, buffer);
}

void DebugUI::Plot(const std::string& name, const float& value, float r_min, float r_max, int buffer = 60)
{
    DebugUI_Group* ui_group = Instance_ptr()->get_group();
    DebugUI_Plot* d_plot;
    if (ui_group->ui_components.find(name) == ui_group->ui_components.end()) {
        d_plot = new DebugUI_Plot(name, buffer, r_min == r_max);
        d_plot->set_range(r_min, r_max);
        ui_group->ui_components[name] = d_plot;
    }
    else {
        d_plot = static_cast<DebugUI_Plot*>(ui_group->ui_components[name]);
    }

    d_plot->add_value(value);
}

void DebugUI::Value(const std::string& name, const float& value)
{
    DebugUI_Group* ui_group = Instance_ptr()->get_group();
    DebugUI_Value* d_value;
    if (ui_group->ui_components.find(name) == ui_group->ui_components.end()) {
        d_value = new DebugUI_Value(name);
        ui_group->ui_components[name] = d_value;
    }
    else {
        d_value = static_cast<DebugUI_Value*>(ui_group->ui_components[name]);
    }
    d_value->set_value(value);
}

void DebugUI::Range(const std::string& name, const float& value) {
    DebugUI_Group* ui_group = Instance_ptr()->get_group();
    DebugUI_Range* d_range;
    if (ui_group->ui_components.find(name) == ui_group->ui_components.end()) {
        d_range = new DebugUI_Range(name);
        ui_group->ui_components[name] = d_range;
    }
    else {
        d_range = static_cast<DebugUI_Range*>(ui_group->ui_components[name]);
    }
    d_range->add_value(value);
}