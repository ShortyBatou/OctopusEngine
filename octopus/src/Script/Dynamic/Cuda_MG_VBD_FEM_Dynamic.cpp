#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_MG_VBD_FEM_Dynamic.h"
#include "Tools/Random.h"
#include <Manager/Debug.h>
#include <set>
#include <GPU/VBD/GPU_MG_VBD.h>
#include <GPU/VBD/GPU_MG_VBD_FEM.h>


GPU_ParticleSystem* Cuda_MG_VBD_FEM_Dynamic::create_particle_system()
{
    return new GPU_MG_VBD(_mesh->geometry(), get_fem_masses(), _iteration, _sub_iterations);
}

void Cuda_MG_VBD_FEM_Dynamic::build_dynamics()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        assert(e == Tetra10 || e == Hexa27);
        GPU_FEM* fem = new GPU_MG_VBD_FEM(e, topo, _mesh->geometry(), _material, _young, _poisson, _damping, _linear, _iteration, _density, _m_distrib, _gpu_ps);
        _gpu_fems[e] = fem;
        _gpu_ps->add_dynamics(fem);
        break;
    }
}