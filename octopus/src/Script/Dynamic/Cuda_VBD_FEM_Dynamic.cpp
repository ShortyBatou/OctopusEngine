#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_VBD_FEM_Dynamic.h"
#include "Tools/Random.h"
#include <Manager/Debug.h>
#include "GPU/PBD/GPU_PBD.h"
#include <set>
#include <Manager/Input.h>


GPU_ParticleSystem* Cuda_VBD_FEM_Dynamic::create_particle_system()
{
    return new GPU_VBD(_mesh->geometry(), get_fem_masses(), _iteration, _sub_iterations, _rho);
}

void Cuda_VBD_FEM_Dynamic::build_dynamics()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        _gpu_ps->add_dynamics(new GPU_VBD_FEM(e, topo, _mesh->geometry(), _material, _young, _poisson, _damping));
        break;
    }
}

void Cuda_VBD_FEM_Dynamic::update() {
    if(Input::Down(Key::Q) || Input::Down(Key::A) || Input::Down(Key::W) || Input::Down(Key::S)) {
        GPU_VBD* vbd = dynamic_cast<GPU_VBD*>(_gpu_ps);
        if(Input::Down(Key::Q)) _sub_iterations++;
        if(Input::Down(Key::A)) _sub_iterations--;

        if(Input::Down(Key::W)) vbd->iteration++;

        if(Input::Down(Key::S)) vbd->iteration--;

        std::cout << _sub_iterations << " " << vbd->iteration << std::endl;
    }

    Cuda_ParticleSystem_Dynamics::update();
}

