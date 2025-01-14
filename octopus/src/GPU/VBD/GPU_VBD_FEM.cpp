#include "GPU/VBD/GPU_VBD_FEM.h"
#include <set>

GPU_VBD_FEM::GPU_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping) :
    GPU_FEM(element, geometry, topology, young, poisson, material), y(nullptr)
{
    const int nb_vertices = static_cast<int>(geometry.size());
    _damping = damping;
    d_owners = new GPU_Owners_Data();

    std::vector<std::vector<int>> e_owners;
    std::vector<std::vector<int>> e_ref_id;
    build_graph_color(topology, nb_vertices, _colors,e_owners,e_ref_id);
    sort_by_color(nb_vertices, _colors, e_owners, e_ref_id);
}

void GPU_VBD_FEM::sort_by_color(const int nb_vertices, const std::vector<int>& colors, const std::vector<std::vector<int>>& e_owners, const std::vector<std::vector<int>>& e_ref_id) const
{
    // sort by color
    std::vector<int> ref_id;
    std::vector<int> owners;
    std::vector<int> nb_owners;
    std::vector<int> owners_offset;
    // sort neighbors
    for(int c = 0; c < d_thread->nb_kernel; ++c) {
        int n_max = 1;
        int nb_vert = 0;
        int n = static_cast<int>(nb_owners.size());
        for(int i = 0; i < nb_vertices; ++i) {
            if(c != colors[i]) continue;
            owners_offset.push_back(static_cast<int>(owners.size()));
            owners.insert(owners.end(), e_owners[i].begin(), e_owners[i].end());
            ref_id.insert(ref_id.end(), e_ref_id[i].begin(), e_ref_id[i].end());
            nb_owners.push_back(static_cast<int>(e_owners[i].size()));

            n_max = std::max(n_max, nb_owners.back());
            nb_vert ++;
        }
        d_thread->grid_size.push_back(nb_vert);
        d_thread->block_size.push_back(n_max * d_fem->nb_quadrature);
        d_thread->nb_threads.push_back(nb_vert * n_max * d_fem->nb_quadrature);
        d_thread->offsets.push_back(n);
    }
    d_owners->cb_nb = new Cuda_Buffer(nb_owners);
    d_owners->cb_eids = new Cuda_Buffer(owners);
    d_owners->cb_offset = new Cuda_Buffer(owners_offset);
    d_owners->cb_ref_vid = new Cuda_Buffer(ref_id);
}


void GPU_VBD_FEM::build_graph_color(const Mesh::Topology &topology, const int nb_vertices,
    std::vector<int> &colors, std::vector<std::vector<int>>& e_neighbors, std::vector<std::vector<int>>& e_ref_id) const
{
    std::vector<std::set<int> > neighbors(nb_vertices);
    e_neighbors.resize(nb_vertices);
    e_ref_id.resize(nb_vertices);
    // for each vertice get all its neighboors
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        int eid = i / d_fem->elem_nb_vert;
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            e_neighbors[topology[i + j]].push_back(eid);
            e_ref_id[topology[i+j]].push_back(j);
            // all vertices inside an element are linked
            for (int k = 0; k < d_fem->elem_nb_vert; ++k) {
                if (k == j) continue;
                neighbors[topology[i + j]].insert(topology[i + k]);
            }
        }
    }
    int max_neighbors = 0;
    d_thread->nb_kernel = 1;
    colors.resize(nb_vertices, -1);
    std::vector<int> available(64, true);
    for (int i = 0; i < nb_vertices; ++i) {
        // for all vertices, check the neighbor elements colors
        max_neighbors = std::max(static_cast<int>(e_neighbors[i].size()), max_neighbors);
        for (const int n: neighbors[i]) {
            if (colors[n] != -1) available[colors[n]] = false;
        }
        for (int c = 0; c < available.size(); ++c) {
            if (available[c]) {
                d_thread->nb_kernel = std::max(d_thread->nb_kernel, c);
                colors[i] = c;
                break;
            }
        }
        std::fill(available.begin(), available.end(), true);
    }
    d_thread->nb_kernel++;
    std::cout << "NB color: " << d_thread->nb_kernel << "  NB neighbors : " << max_neighbors << std::endl;
}
