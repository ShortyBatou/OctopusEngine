#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_Mixed_VBD_FEM_Dynamic.h"
#include "Tools/Random.h"
#include <Manager/Debug.h>
#include <set>
#include <GPU/VBD/GPU_Mixed_VBD.h>
#include <Manager/Input.h>


GPU_ParticleSystem* Cuda_Mixed_VBD_FEM_Dynamic::create_particle_system()
{
    return new GPU_Mixed_VBD(_mesh->geometry(), get_fem_masses(), _iteration, _sub_iterations, _exp_it);
}

void Cuda_Mixed_VBD_FEM_Dynamic::build_dynamics()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        GPU_FEM* fem = new GPU_Mixed_VBD_FEM(e, topo, _mesh->geometry(), _material, _young, _poisson, _damping);
        _gpu_fems[e] = fem;
        _gpu_ps->add_dynamics(fem);
        break;
    }
}