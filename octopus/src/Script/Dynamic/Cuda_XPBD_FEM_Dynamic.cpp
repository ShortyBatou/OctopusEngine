#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_XPBD_FEM_Dynamic.h"
#include "Tools/Random.h"
#include <Manager/Debug.h>
#include "GPU/PBD/GPU_PBD.h"
#include "GPU/PBD/GPU_PBD_FEM.h"
#include "GPU/PBD/GPU_PBD_FEM_Coupled.h"
#include <random>
#include <set>
#include <Manager/Input.h>


GPU_Integrator* Cuda_XPBD_FEM_Dynamic::create_integrator()
{
   return new GPU_PBD(_damping);
}

void Cuda_XPBD_FEM_Dynamic::build_dynamics()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;

        if(_coupled_fem)
            _gpu_xpbd_fems[e] = new GPU_PBD_FEM_Coupled(e, _mesh->geometry(), topo, _young, _poisson, _material);
        else
        {
            _gpu_xpbd_fems[e] = new GPU_PBD_FEM(e, _mesh->geometry(), topo, _young, _poisson, _material, true);
        }

        _gpu_fems[e] = _gpu_xpbd_fems[e];
        // create CUDA PBD Gauss-Seidel
        _gpu_integrator->add_dynamics(_gpu_fems[e]);
    }
}


void Cuda_XPBD_FEM_Dynamic::update() {
    if(Time::Frame() == 1) {
        // coloration
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            const int nb_color = static_cast<int>(_gpu_xpbd_fems[e]->colors.size());
            std::vector<Color> color_map(nb_color);
            for(auto & c : color_map) {
                c = ColorBase::HSL2RGB(Random::Range(0.f,360.f), Random::Range(42.f,98.f), Random::Range(40.f,90.f));
            }
            _display_colors[e].resize(nb_color);
            for(int i = 0; i < nb_color; ++i) {
                _display_colors[e][i] = color_map[_gpu_xpbd_fems[e]->colors[i]];
            }
        }
    }

    if(Input::Loop(Key::KP_1)) {
        GL_Graphic* graphic = entity()->get_component<GL_Graphic>();
        for(auto&[e, topo] : _mesh->topologies()) {
            if(topo.empty()) continue;
            graphic->set_ecolors(e, _display_colors[e]);
            graphic->set_multi_color(true);
            graphic->set_element_color(true);
        }
    }
    else {
        GL_Graphic* graphic = entity()->get_component<GL_Graphic>();
        graphic->set_multi_color(false);
        graphic->set_element_color(false);
    }
    if(Input::Down(Key::A)) _sub_iterations++;
    if(Input::Down(Key::Q)) _sub_iterations--;
    if(Input::Down(Key::A) || Input::Down(Key::Q)) std::cout << _sub_iterations << std::endl;
    Cuda_ParticleSystem_Dynamics::update();
}

