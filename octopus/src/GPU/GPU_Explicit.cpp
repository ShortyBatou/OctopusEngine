#include "GPU/GPU_Explicit.h"

void GPU_Explicit::step(const scalar dt) const
{
    for (GPU_Dynamic* dynamic : _dynamics)
        dynamic->step(this, dt);

    _integrator->step(this, dt);
}

GPU_Explicit_FEM::GPU_Explicit_FEM(const Element element, const Mesh::Geometry& geometry, const Mesh::Topology& topology, // mesh
                                   const scalar young, const scalar poisson, const Material material, const scalar damping)
    : GPU_FEM(element, geometry, topology, young, poisson, material), _damping(damping)
{
    cb_topology = new Cuda_Buffer(topology);
    int nb_vertices = geometry.size();
    std::vector<std::vector<int>> e_neighbors(nb_vertices);
    std::vector<std::vector<int>> e_ref_id(nb_vertices);
    // for each vertice get all its neighboors
    for (int i = 0; i < topology.size(); i += elem_nb_vert) {
        for (int j = 0; j < elem_nb_vert; ++j) {
            e_neighbors[topology[i + j]].push_back(i / elem_nb_vert);
            e_ref_id[topology[i+j]].push_back(j);
        }
    }
    // sort by color
    std::vector<int> ref_id;
    std::vector<int> neighbors;
    std::vector<int> nb_neighbors;
    std::vector<int> neighbors_offset;

    // sort neighbors
    int n_max = 1;

    const int n = static_cast<int>(nb_neighbors.size());
    for(int i = 0; i < nb_vertices; ++i) {
        neighbors_offset.push_back(static_cast<int>(neighbors.size()));
        neighbors.insert(neighbors.end(), e_neighbors[i].begin(), e_neighbors[i].end());
        ref_id.insert(ref_id.end(), e_ref_id[i].begin(), e_ref_id[i].end());
        nb_neighbors.push_back(static_cast<int>(e_neighbors[i].size()));
        n_max = std::max(n_max, nb_neighbors.back());
    }

    _block_size = n_max * nb_quadrature;
    _nb_threads = nb_vertices * _block_size;
    c_offsets.push_back(n);

    cb_nb_neighbors = new Cuda_Buffer(nb_neighbors);
    cb_neighbors = new Cuda_Buffer(neighbors);
    cb_neighbors_offset = new Cuda_Buffer(neighbors_offset);
    cb_ref_vid = new Cuda_Buffer(ref_id);
}