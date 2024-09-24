#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_VBD_FEM_Dynamic.h"
#include "Tools/Random.h"
#include <Manager/Debug.h>
#include "GPU/GPU_PBD.h"
#include <set>


void Cuda_VBD_FEM_Dynamic::init() {
    _mesh = _entity->get_component<Mesh>();
    std::vector masses(_mesh->nb_vertices(),0.f);
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        // récupérer la masse
        const std::vector<scalar> e_masses = compute_fem_mass(e, _mesh->geometry(),topo, _density); // depends on density
        for(size_t i = 0; i < e_masses.size(); i++)
            masses[i] += e_masses[i];
    }

    vbd = new GPU_VBD(_mesh->geometry(), masses, _iteration, _sub_iteration, 0.f);
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        vbd->dynamic = new GPU_VBD_FEM(e, topo, _mesh->geometry(), _young, _poisson);
        break;
    }

}

void Cuda_VBD_FEM_Dynamic::update() {
    /*if(Time::Frame() == 1) {
        // coloration
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            std::vector<int> colors;
            const int nb_color = get_vertex_coloration(e, _mesh->nb_vertices(), topo, colors);
            std::vector<Color> color_map(nb_color);
            for(auto & c : color_map) {
                c = ColorBase::HSL2RGB(Random::Range(0.f,360.f), Random::Range(42.f,98.f), Random::Range(40.f,90.f));
            }
            _display_colors[e].resize(colors.size());
            for(int i = 0; i < colors.size(); ++i) {
                _display_colors[e][i] = color_map[colors[i]];
            }
            break;
        }
    }

    if(Time::Frame() > 0) {

        GL_Graphic* graphic = entity()->get_component<GL_Graphic>();
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            graphic->set_vcolors(_display_colors[e]);
            graphic->set_multi_color(true);
            break;
        }

    }

    Time::Tic();


    const scalar time = Time::Tac() * 1000.f;
    DebugUI::Begin("VBD");
    DebugUI::Plot("Time GPU VBD ", time, 600);
    DebugUI::Value("Time", time);
    DebugUI::Range("Range", time);
    DebugUI::End();*/

    vbd->step(Time::Fixed_DeltaTime());
    vbd->get_position(_mesh->geometry());
}

