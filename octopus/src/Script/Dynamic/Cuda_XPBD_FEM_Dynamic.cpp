#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_XPBD_FEM_Dynamic.h"
#include "Tools/Random.h"
#include <Manager/Debug.h>
#include "GPU/GPU_PBD.h"
#include <random>
#include <set>



void Cuda_XPBD_FEM_Dynamic::init() {
    _mesh = _entity->get_component<Mesh>();
    std::vector masses(_mesh->nb_vertices(),0.f);
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        // récupérer la masse
        const std::vector<scalar> e_masses = compute_fem_mass(e, _mesh->geometry(),topo, _density); // depends on density
        for(size_t i = 0; i < e_masses.size(); i++)
            masses[i] += e_masses[i];
    }

    _gpu_pbd = new GPU_PBD(_mesh->geometry(), masses, _iteration, _damping);
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        _gpu_fems[e] = new GPU_PBD_FEM(e, _mesh->geometry(), topo, _young, _poisson);

        // récupérer la masse
        const std::vector<scalar> e_masses = compute_fem_mass(e, _mesh->geometry(),topo, _density); // depends on density
        for(size_t i = 0; i < e_masses.size(); i++)
            masses[i] += e_masses[i];

        // create CUDA PBD Gauss-Seidel
        _gpu_pbd->dynamic.push_back(_gpu_fems[e]);
    }
}

void Cuda_XPBD_FEM_Dynamic::update() {
    if(Time::Frame() == 1) {
        // coloration
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            const int nb_color = static_cast<int>(_gpu_fems[e]->colors.size());
            std::vector<Color> color_map(nb_color);
            for(auto & c : color_map) {
                c = ColorBase::HSL2RGB(Random::Range(0.f,360.f), Random::Range(42.f,98.f), Random::Range(40.f,90.f));
            }
            _display_colors[e].resize(nb_color);
            for(int i = 0; i < nb_color; ++i) {
                _display_colors[e][i] = color_map[_gpu_fems[e]->colors[i]];
            }

        }
    }

    if(Time::Frame() > 0) {

        GL_Graphic* graphic = entity()->get_component<GL_Graphic>();
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            graphic->set_ecolors(e, _display_colors[e]);
            graphic->set_multi_color(true);
            graphic->set_element_color(true);
        }

    }

    Time::Tic();
    _gpu_pbd->step(Time::Fixed_DeltaTime());
    _gpu_pbd->get_position(_mesh->geometry());

    const scalar time = Time::Tac() * 1000.f;
    DebugUI::Begin("XPBD");
    DebugUI::Plot("Time GPU XPBD ", time, 600);
    DebugUI::Value("Time", time);
    DebugUI::Range("Range", time);
    DebugUI::End();
}

