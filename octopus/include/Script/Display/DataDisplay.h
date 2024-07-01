#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"
#include "Rendering/GL_Graphic.h"
#include "Script/Dynamic/FEM_Dynamic.h"
#include "Manager/Input.h"



struct FEM_DataDisplay : public Component {
    enum Type {
        BaseColor, Displacement, Stress, V_Stress, Volume, Volume_Diff, Mass, Inv_Mass, Velocity, None
    };

    FEM_DataDisplay(Type mode = BaseColor, ColorMap::Type type = ColorMap::Type::Default) : _mode(mode), color_map(type) {}

    static char* Type_To_Str(Type mode) {
        switch (mode)
        {
        case BaseColor: return "BaseColor";
        case Displacement: return "Displacement";
        case Stress: return "Stress";
        case V_Stress: return "Vertice Stress";
        case Volume: return "Volume";
        case Volume_Diff: return "Volume Diff";
        case Mass: return "Mass";
        case Inv_Mass: return "Inverse Mass";
        case Velocity: return "Velocity";
        default: return "";
        }
    }

    virtual void init() override {
        _fem_dynamic = entity()->get_component<FEM_Dynamic>();
        _mesh = entity()->get_component<Mesh>();
        assert(_fem_dynamic);
    }


    std::vector<Color> convert_to_color(std::vector<scalar>& data) {
        scalar min = *std::min_element(data.begin(), data.end());
        scalar max = *std::max_element(data.begin(), data.end());
        scalar diff = max - min;
        
        std::vector<Color> colors(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            scalar t = (diff > eps) ? (data[i] - min) / diff : 0.f;
            colors[i] = ColorMap::evaluate(t);
        }
        return colors;
    }

    virtual void update() override {
        _graphic = entity()->get_component<GL_Graphic>();
        // compute data
        auto ps = _fem_dynamic->getParticleSystem();

        ColorMap::Set_Type(color_map);
        if (_mode == BaseColor) {
            _graphic->set_multi_color(false);
            _graphic->set_element_color(false);
        }
        else if (_mode == V_Stress)
        {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(false);
            std::vector<scalar> data = _fem_dynamic->get_stress_vertices();
            _graphic->set_vcolors(convert_to_color(data));
        }
        else if (_mode == Displacement) {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(false);
            std::vector<scalar> data = _fem_dynamic->get_displacement_norm();
            _graphic->set_vcolors(convert_to_color(data));
        }
        else if (_mode == Mass) {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(false);
            std::vector<scalar> data = _fem_dynamic->get_masses();
            _graphic->set_vcolors(convert_to_color(data));
        }
        else if (_mode == Inv_Mass) {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(false);
            std::vector<scalar> data = _fem_dynamic->get_massses_inv();
            _graphic->set_vcolors(convert_to_color(data));
        }
        else if (_mode == Velocity) {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(false);
            std::vector<scalar> data = _fem_dynamic->get_velocity_norm();
            _graphic->set_vcolors(convert_to_color(data));
        }
        else if (_mode == Stress) {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(true);
            std::map <Element, std::vector<scalar>> e_data = _fem_dynamic->get_stress();
            for (auto& it : e_data) {
                Element type = it.first;
                std::vector<scalar>& data = it.second;
                _graphic->set_ecolors(type, convert_to_color(data));
            }
        }
        else if (_mode == Volume) {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(true);
            std::map <Element, std::vector<scalar>> e_data = _fem_dynamic->get_volume();
            for (auto& it : e_data) {
                Element type = it.first;
                std::vector<scalar>& data = it.second;
                _graphic->set_ecolors(type, convert_to_color(data));
            }
        }
        else if (_mode == Volume_Diff) {
            _graphic->set_multi_color(true);
            _graphic->set_element_color(true);
            std::map <Element, std::vector<scalar>> e_data = _fem_dynamic->get_volume_diff();
            for (auto& it : e_data) {
                Element type = it.first;
                std::vector<scalar>& data = it.second;
                _graphic->set_ecolors(type, convert_to_color(data));
            }
        }
    }


    ColorMap::Type color_map;
    Type _mode;
private:
    Mesh* _mesh;
    GL_Graphic* _graphic;
    FEM_Dynamic* _fem_dynamic;
};