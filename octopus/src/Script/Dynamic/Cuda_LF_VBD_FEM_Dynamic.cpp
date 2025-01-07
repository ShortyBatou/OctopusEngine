#include "Core/Entity.h"
#include "Script/Dynamic/Cuda_LF_VBD_FEM_Dynamic.h"
#include <GPU/VBD/GPU_VBD.h>
#include <GPU/VBD/GPU_LF_VBD_FEM.h>

void Cuda_LF_VBD_FEM_Dynamic::build_dynamics()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        GPU_FEM* fem = new GPU_LF_VBD_FEM(e, topo, _mesh->geometry(), _material, _young, _poisson, _damping);
        _gpu_fems[e] = fem;
        _gpu_ps->add_dynamics(fem);
        break;
    }
}