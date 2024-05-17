#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"
#include "Rendering/GL_Graphic.h"
#include "Script/Dynamic/FEM_Dynamic.h"

enum FEM_Data {
    BaseColor, Displacement, Stress, Volume
};

struct FEM_DataDisplay : public Component {
    FEM_DataDisplay(FEM_Data mode = BaseColor, ColorMap::Type type = ColorMap::Type::Default) : _mode(mode), color_map(type) {}

    virtual void init() override {
        _graphic = entity()->get_component<GL_Graphic>();
        _fem_dynamic = entity()->get_component<FEM_Dynamic>();
        _mesh = entity()->get_component<Mesh>();
        assert(_graphic && _fem_dynamic);
    }

    virtual void update() override {
        // compute data
        auto ps = _fem_dynamic->getParticleSystem();
        //std::vector<scalar> data(ps->nb_particles());
        //for (size_t i = 0; i < data.size(); ++i) {
        //    Particle* p = ps->get(i);
        //    data[i] = glm::length(p->position - p->init_position);
        //}

        std::vector<scalar> data = _fem_dynamic->get_volume_diff();

        // min max data for color interpolation
        scalar min = 0.f;
        scalar max = *std::max_element(data.begin(), data.end());
        std::vector<Color> colors(data.size());
        
        ColorMap::Set_Type(color_map);
        for (size_t i = 0; i < data.size(); ++i) {
            scalar t = (data[i] - min) / scalar(max - min);
            colors[i] = ColorMap::evaluate(t);
        }
       
        _graphic->set_multi_color(true);
        _graphic->set_element_color(true);
        _graphic->set_ecolors(Tetra, colors);
        //_graphic->set_vcolors(colors);
    }


public:
    ColorMap::Type color_map;

private:
    Mesh* _mesh;
    GL_Graphic* _graphic;
    FEM_Dynamic* _fem_dynamic;
    FEM_Data _mode;
};