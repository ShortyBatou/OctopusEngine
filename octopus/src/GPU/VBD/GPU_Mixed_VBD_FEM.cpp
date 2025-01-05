#include "GPU/VBD/GPU_Mixed_VBD_FEM.h"
#include <set>

GPU_Mixed_VBD_FEM::GPU_Mixed_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping) :
    GPU_VBD_FEM(element, topology, geometry, material, young, poisson, damping)
{

    p_forces = new Cuda_Buffer<Vector3>(std::vector<Vector3>(topology.size()));
    d_exp_thread = new Thread_Data();
    int block_size = 0;
    for(int i = 0; i < d_thread->nb_kernel; ++i)
    {
        block_size = std::max(block_size, d_thread->block_size[i]);
    }
    d_exp_thread->nb_kernel = 1;
    d_exp_thread->block_size.push_back(block_size);
    d_exp_thread->nb_threads.push_back(static_cast<int>(geometry.size()) * block_size);
    d_exp_thread->grid_size.push_back((d_exp_thread->nb_threads[0] + block_size-1) / block_size);
    d_exp_thread->offsets.push_back(0);
}