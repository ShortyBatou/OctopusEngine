#include "GPU/VBD/GPU_MG_VBD_FEM.h"
#include <random>
#include <numeric>

__global__ void kernel_prolongation_tetra(GPU_ParticleSystem_Parameters ps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;

}


void GPU_MG_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt)
{
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937());

    for (const int c : kernels)
    {

    }
}
