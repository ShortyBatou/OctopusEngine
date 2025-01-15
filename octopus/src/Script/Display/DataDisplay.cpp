#include "Script/Display/DataDisplay.h"

std::string FEM_DataDisplay::Type_To_Str(const Type mode) {
    switch (mode) {
        case BaseColor: return "BaseColor";
        case Displacement: return "Displacement";
        case Stress: return "Stress";
        case V_Stress: return "Vertice Stress";
        case Volume: return "Volume";
        case Volume_Diff: return "Volume Diff";
        case Mass: return "Mass";
        case Inv_Mass: return "Inverse Mass";
        case Velocity: return "Velocity";
        case Mask: return "Mask";
        default: return "";
    }
}

void FEM_DataDisplay::init() {
    _ps_dynamic = entity()->get_component<ParticleSystemDynamics_Getters>();
    _fem_dynamic = entity()->get_component<FEM_Dynamic_Getters>();
    _mesh = entity()->get_component<Mesh>();
    assert(_fem_dynamic);
    assert(_ps_dynamic);
}


std::vector<Color> FEM_DataDisplay::convert_to_color(const std::vector<scalar> &data, const  scalar vmin, const scalar vmax) {
    const scalar diff = vmax - vmin;
    std::vector<Color> colors(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        const scalar t = (diff > eps) ? (data[i] - vmin) / diff : 0.f;
        colors[i] = ColorMap::evaluate(t);
    }
    return colors;
}


std::vector<Color> FEM_DataDisplay::convert_to_color(const std::vector<scalar> &data) {
    const scalar min = *std::min_element(data.begin(), data.end());
    const scalar max = *std::max_element(data.begin(), data.end());
    const scalar diff = max - min;

    std::vector<Color> colors(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        const scalar t = (diff > eps) ? (data[i] - min) / diff : 0.f;
        colors[i] = ColorMap::evaluate(t);
    }
    return colors;
}

void FEM_DataDisplay::update() {
    _graphic = entity()->get_component<GL_Graphic>();
    // compute data
    ColorMap::Set_Type(color_map);
    if (_mode == BaseColor) {
        _graphic->set_multi_color(false);
        _graphic->set_element_color(false);
    } else if (_mode == V_Stress) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(false);
        const std::vector<scalar> data = _fem_dynamic->get_stress_vertices();
        _graphic->set_vcolors(convert_to_color(data));
    } else if (_mode == Displacement) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(false);
        const std::vector<scalar> data = _ps_dynamic->get_displacement_norm();
        _graphic->set_vcolors(convert_to_color(data));
    } else if (_mode == Mass) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(false);
        const std::vector<scalar> data = _ps_dynamic->get_masses();
        _graphic->set_vcolors(convert_to_color(data));
    } else if (_mode == Inv_Mass) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(false);
        const std::vector<scalar> data = _ps_dynamic->get_massses_inv();
        _graphic->set_vcolors(convert_to_color(data));
    } else if (_mode == Velocity) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(false);
        const std::vector<scalar> data = _ps_dynamic->get_velocity_norm();
        _graphic->set_vcolors(convert_to_color(data));
    } else if (_mode == Mask) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(false);
        std::vector<int> i_data =_ps_dynamic->get_masks();
        const std::vector<scalar> data(i_data.begin(), i_data.end());
        _graphic->set_vcolors(convert_to_color(data,0,3));
    } else if (_mode == Stress) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(true);
        std::map<Element, std::vector<scalar> > e_data = _fem_dynamic->get_stress();
        for (auto &[e, data]: e_data) {
            _graphic->set_ecolors(e, convert_to_color(data));
        }
    } else if (_mode == Volume) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(true);
        std::map<Element, std::vector<scalar> > e_data = _fem_dynamic->get_volume();
        for (auto &[e, data]: e_data) {
            _graphic->set_ecolors(e, convert_to_color(data));
        }
    } else if (_mode == Volume_Diff) {
        _graphic->set_multi_color(true);
        _graphic->set_element_color(true);
        std::map<Element, std::vector<scalar> > e_data = _fem_dynamic->get_volume_diff();
        for (auto &[e, data]: e_data) {
            _graphic->set_ecolors(e, convert_to_color(data));
        }
    }
}
