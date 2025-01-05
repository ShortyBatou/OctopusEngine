#include "GPU/Explicit/GPU_Explicit.h"

void GPU_Explicit::step(scalar dt)
{
    for (GPU_Dynamic* dynamic : _dynamics)
        dynamic->step(this, dt);

    _integrator->step(this, dt);
}

GPU_Explicit_FEM::GPU_Explicit_FEM(const Element element, const Mesh::Geometry& geometry, const Mesh::Topology& topology, // mesh
                                   const scalar young, const scalar poisson, const Material material, const scalar damping)
    : GPU_FEM(element, geometry, topology, young, poisson, material), _damping(damping)
{
    const int nb_vertices = static_cast<int>(geometry.size());
    std::vector<std::vector<int>> e_owners(nb_vertices);
    std::vector<std::vector<int>> e_ref_id(nb_vertices);
    // for each vertice get all its neighboors
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            e_owners[topology[i + j]].push_back(i / d_fem->elem_nb_vert);
            e_ref_id[topology[i+j]].push_back(j);
        }
    }
    // sort by color
    std::vector<int> ref_id;
    std::vector<int> owners;
    std::vector<int> nb_owners;
    std::vector<int> offset;

    // sort neighbors
    int n_max = 1;

    for(int i = 0; i < nb_vertices; ++i) {
        offset.push_back(static_cast<int>(owners.size()));
        owners.insert(owners.end(), e_owners[i].begin(), e_owners[i].end());
        ref_id.insert(ref_id.end(), e_ref_id[i].begin(), e_ref_id[i].end());
        nb_owners.push_back(static_cast<int>(e_owners[i].size()));
        n_max = std::max(n_max, nb_owners.back());
    }

    d_thread->block_size.push_back(n_max * d_fem->nb_quadrature);
    d_thread->nb_threads.push_back(nb_vertices * d_thread->block_size[0]);
    d_thread->grid_size.push_back(nb_vertices);
    d_thread->offsets.push_back(0);
    std::cout << "EXPLICIT NB THREAD = " << nb_vertices * d_thread->block_size[0] << std::endl;
    d_owners = new GPU_Owners_Data();
    d_owners->cb_nb = new Cuda_Buffer(nb_owners);
    d_owners->cb_eids = new Cuda_Buffer(owners);
    d_owners->cb_offset = new Cuda_Buffer(offset);
    d_owners->cb_ref_vid = new Cuda_Buffer(ref_id);
}