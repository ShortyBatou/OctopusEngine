#include "GPU/VBD/GPU_Mixed_VBD_FEM.h"
#include <GPU/Explicit/GPU_Explicit.h>


void GPU_Mixed_VBD_FEM::explicit_step(GPU_ParticleSystem* ps, scalar dt) const
{
    kernel_explicit_fem_eval_force<<<d_exp_thread->grid_size[0], d_exp_thread->block_size[0]>>>(
        d_exp_thread->nb_threads[0], _damping, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
    );
}
