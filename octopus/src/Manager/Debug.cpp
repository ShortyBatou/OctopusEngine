#pragma once
#include "Manager/Debug.h"
#include "imgui.h"
#include <unordered_map>
#include "UI/AppInfo.h"
#include <sstream>

void Debug::add_vertice(const Vector3 &p) {
    assert(_mesh != nullptr);
    assert(_graphic != nullptr);
    _mesh->geometry().push_back(p);
    _graphic->vcolors().push_back(_current_color);
}

void Debug::add_line(int a, int b) {
    assert(_mesh != nullptr);
    _mesh->topology(Element::Line).push_back(a);
    _mesh->topology(Element::Line).push_back(b);
}

void Debug::clear() const {
    _mesh->clear();
    _graphic->vcolors().clear();
}


void Debug::SetColor(const Color &color) {
    Instance().set_current_color(color);
}

void Debug::Line(const Vector3 &a, const Vector3 &b) {
    Debug *debug = Instance_ptr();
    int _i_start = debug->_mesh->nb_vertices();
    debug->add_vertice(a);
    debug->add_vertice(b);
    debug->add_line(_i_start, _i_start + 1);
}

void Debug::Vector(const Vector3 &p, const Vector3 &direction, scalar length = 1.) {
    Line(p, p + direction * length);
}

void Debug::UnitGrid(int _size) {
    _size = std::max(0, _size);
    int nb_square = (_size + 1) * 2;
    Vector3 o(-static_cast<scalar>(_size + 1), 0., -static_cast<scalar>(_size + 1u));
    const Vector3 dir_x = Unit3D::right() * static_cast<scalar>(nb_square);
    const Vector3 dir_z = Unit3D::forward() * static_cast<scalar>(nb_square);
    for (int i = 0; i <= nb_square; ++i) {
        Vector(o + Unit3D::right() * static_cast<scalar>(i), dir_z);
        Vector(o + Unit3D::forward() * static_cast<scalar>(i), dir_x);
    }
}

void Debug::Axis(const Vector3 &p, const Matrix4x4 &rot, scalar length = 1.) {
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

void Debug::Axis(const Vector3 &p, scalar length = 1.) {
    Debug::Axis(p, Matrix::Identity4x4(), length);
}

void Debug::Cube(const Vector3 &p_min, const Vector3 &p_max) {
    Debug *debug = Instance_ptr();
    int _i_start = debug->_mesh->nb_vertices();
    for (int i = 0; i < 8; ++i) {
        Vector3 v(p_min);
        if (i & 1) v.x = p_max.x;
        if (i & 2) v.y = p_max.y;
        if (i & 4) v.z = p_max.z;
        debug->add_vertice(v);
    }
    static int box_topo[24] = {
        0, 1, 1, 3, 3, 2, 2, 0, 2, 6, 6, 7,
        7, 3, 7, 5, 5, 1, 6, 4, 4, 0, 4, 5
    };
    for (int i = 0; i < 24; i += 2) {
        debug->add_line(_i_start + box_topo[i], _i_start + box_topo[i + 1]);
    }
}

void Debug::Cube(const Vector3 &p = Unit3D::Zero(), scalar size = scalar(1.f)) {
    Vector3 p_min(p - Vector3(size) * scalar(0.5));
    Vector3 p_max(p + Vector3(size) * scalar(0.5));
    Debug::Cube(p_min, p_max);
}


/// MUST BE ELSEWHERE !
std::string DebugUI_Component::convert_scientific(const double value, const int precision) {
    std::ostringstream str;
    str << std::scientific << std::setprecision(precision) << value;
    std::string s = str.str();
    return s;
}


void DebugUI_Group::draw(const int w, const int h) {
    if (ImGui::TreeNodeEx(this->name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
        for (auto ui: ui_components) {
            ui.second->draw(w, h);
        }
        ImGui::TreePop();
        ImGui::Spacing();
    }
}

DebugUI_Group::~DebugUI_Group() {
    for (const auto &ui: ui_components) {
        delete ui.second;
    }
    ui_components.clear();
}


void DebugUI_Plot::draw(const int w, const int h) {
    ImGui::PlotLines("", _values.data(), _size, _offset, "", _r_min, _r_max, ImVec2(static_cast<scalar>(w) * 0.8f, 60));
}

void DebugUI_Plot::add_value(float value) {
    _values[_offset] = value;
    _offset = (_offset + 1) % _size;
    if (_auto_range) {
        _r_min = std::min(value, _r_min);
        _r_max = std::max(value, _r_max);
    }
}


void DebugUI_Range::draw(int w, int h) {
    std::string s_min = convert_scientific(_vmin);
    std::string s_max = convert_scientific(_vmax);
    std::string s_mean = convert_scientific(_vmean);
    ImGui::Text("Min = %s   Max = %s   Mean = %s", s_min.c_str(), s_max.c_str(), s_mean.c_str());
}

void DebugUI_Range::add_value(float value) {
    ++_nb_values;
    _vsum += value;
    _vmean = _vsum / static_cast<scalar>(_nb_values);
    _vmin = std::min(value, _vmin);
    _vmax = std::max(value, _vmax);
}


void DebugUI_Value::draw(int w, int h) {
    std::string s_val = convert_scientific(_value);
    ImGui::Text("%s = %s", this->name.c_str(), s_val.c_str());
}


void DebugUI::draw() const {
    int w, h;
    AppInfo::Window_sizes(w, h);
    ImGui::SetNextWindowPos(ImVec2(static_cast<scalar>(w) - 410.f, 10.f));
    int sx = 400;
    int sy = std::max(h - 10, 10);
    ImGui::SetNextWindowSize(ImVec2(scalar(sx), scalar(sy)));
    ImGui::Begin("Debug");
    for (auto ui: ui_groups) {
        ui.second->draw(sx, sy);
    }

    ImGui::End();
}


DebugUI::~DebugUI() {
    for (const auto &c: ui_groups) {
        delete c.second;
    }
    ui_groups.clear();
}


void DebugUI::Begin(const std::string &name) {
    DebugUI *ui_debug = Instance_ptr();
    if (ui_debug->ui_groups.find(name) == ui_debug->ui_groups.end()) {
        ui_debug->ui_groups[name] = new DebugUI_Group(name);
    }
    Instance_ptr()->current_group = name;
}

void DebugUI::End() {
    Instance_ptr()->current_group = "";
}

void DebugUI::Plot(const std::string &name, const float &value, int buffer) {
    Plot(name, value, 0, 0, buffer);
}

void DebugUI::Plot(const std::string &name, const float &value, float r_min, float r_max, int buffer) {
    DebugUI_Group *ui_group = Instance_ptr()->get_group();
    DebugUI_Plot *d_plot;
    if (ui_group->ui_components.find(name) == ui_group->ui_components.end()) {
        d_plot = new DebugUI_Plot(name, buffer, r_min == r_max);
        d_plot->set_range(r_min, r_max);
        ui_group->ui_components[name] = d_plot;
    } else {
        d_plot = dynamic_cast<DebugUI_Plot *>(ui_group->ui_components[name]);
    }

    d_plot->add_value(value);
}

void DebugUI::Value(const std::string &name, const float &value) {
    DebugUI_Group *ui_group = Instance_ptr()->get_group();
    DebugUI_Value *d_value;
    if (ui_group->ui_components.find(name) == ui_group->ui_components.end()) {
        d_value = new DebugUI_Value(name);
        ui_group->ui_components[name] = d_value;
    } else {
        d_value = dynamic_cast<DebugUI_Value *>(ui_group->ui_components[name]);
    }
    d_value->set_value(value);
}

void DebugUI::Range(const std::string &name, const float &value) {
    DebugUI_Group *ui_group = Instance_ptr()->get_group();
    DebugUI_Range *d_range;
    if (ui_group->ui_components.find(name) == ui_group->ui_components.end()) {
        d_range = new DebugUI_Range(name);
        ui_group->ui_components[name] = d_range;
    } else {
        d_range = dynamic_cast<DebugUI_Range *>(ui_group->ui_components[name]);
    }
    d_range->add_value(value);
}
