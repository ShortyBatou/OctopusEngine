#include "GPU/VBD/GPU_MG_VBD_FEM.h"
#include <random>
#include <numeric>

__global__ void kernel_prolongation(const int n, GPU_ParticleSystem_Parameters ps, GPU_MG_Interpolation_Parameters inter) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int* primitive = inter.primitives + tid * inter.nb_vert_primitives;
    Vector3 dt_p = Vector3(0);
    for(int i = 0; i < inter.nb_vert_primitives; ++i)
    {
        const int vid = primitive[i];
        dt_p += ps.p[vid] - ps.last_p[vid];
    }
    ps.p[inter.ids[tid]] = ps.last_p[inter.ids[tid]] + dt_p * inter.weight;
}


void GPU_MG_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt)
{
    const auto ps_param = ps->get_parameters();

    it_count++;
    const int last_level = level;
    while(it_count > nb_iterations[level])
    {
        it_count = 1;
        level = (level + 1) % static_cast<int>(nb_iterations.size());
    }

    if (last_level != level && level == 1)
    {
        for(int i = 0; i < interpolations.size(); ++i)
        {
            const auto inter_param = get_interpolation_parameters(i);
            kernel_prolongation<<<(inter_param.nb_ids+31)/32,32>>>(inter_param.nb_ids, ps_param, inter_param);
        }
    }
    ps->_data->_cb_mass = masses[level];
    d_thread = l_threads[level];
    d_fem = l_fems[level];
    d_owners = l_owners[level];

    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937());

    const auto fem_param = get_fem_parameters();
    const auto owners_param = get_owners_parameters();
    for (const int c : kernels)
    {
        kernel_vbd_solve<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
            d_thread->nb_threads[c], _damping, dt, d_thread->offsets[c],
             y->buffer, *d_material, ps_param, fem_param, owners_param
        );
    }

    if (level == 1)
    {
        for(int i = 0; i < interpolations.size(); ++i)
        {
            const auto inter_param = get_interpolation_parameters(i);
            kernel_prolongation<<<(inter_param.nb_ids+31)/32,32>>>(inter_param.nb_ids, ps_param, inter_param);
        }
    }
}
