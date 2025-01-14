#include "GPU/PBD/GPU_PBD_FEM.h"
#include <set>

// MUST BE ELSEWERE
void GPU_PBD_FEM::build_graph_color(const Mesh::Topology &topology, const int nb_vert, std::vector<int>& colors) const {
    d_thread->nb_kernel = 1;
    std::vector<std::set<int> > owners(nb_vert);
    // for each vertice get elements that own this vertice
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            owners[topology[i + j]].insert(i / d_fem->elem_nb_vert);
        }
    }

    colors.resize(topology.size() / d_fem->elem_nb_vert, -1);
    std::vector<int> available(64, true);
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        // for all vertices, check the neighbor elements colors
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            for (const int& n: owners[topology[i + j]]) {
                if (colors[n] != -1) available[colors[n]] = false;
            }
        }
        for (int c = 0; c < available.size(); ++c) {
            if (available[c]) {
                d_thread->nb_kernel = std::max(d_thread->nb_kernel, c);
                colors[i / d_fem->elem_nb_vert] = c;
                break;
            }
        }
        std::fill(available.begin(), available.end(), true);
    }
    d_thread->nb_kernel++;
    std::cout << "NB color: " << d_thread->nb_kernel << std::endl;

}

void GPU_PBD_FEM::build_thread_by_color(const std::vector<int>& colors) {
    // sort element by color and get color group sizes
    d_thread->offsets.resize(d_thread->nb_kernel);
    std::vector<int> eids;
    int count = 0;
    for (int c = 0; c < d_thread->nb_kernel; ++c) {
        d_thread->offsets[c] = count;
        for (int i = 0; i < d_fem->nb_element; ++i) {
            if (colors[i] != c) continue;
            eids.push_back(i);
            count++;
        }
    }

    const int s_off = static_cast<int>(d_thread->offsets.size());
    const int s_eids = static_cast<int>(eids.size());
    // build constant value for FEM simulation and init buffers
    for (int i = 0; i < s_off; ++i) {
        const int nb = (i < s_off - 1 ? d_thread->offsets[i + 1] : s_eids) - d_thread->offsets[i];
        d_thread->nb_threads.push_back(nb);
        d_thread->block_size.push_back(d_fem->nb_quadrature);
        d_thread->grid_size.push_back((nb+d_fem->nb_quadrature-1) / d_fem->nb_quadrature);
    }

    cb_eid = new Cuda_Buffer(eids);
}


GPU_PBD_FEM::GPU_PBD_FEM(const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                         const scalar young, const scalar poisson, const Material material)
        : GPU_FEM(element, geometry, topology, young, poisson, material), cb_eid(nullptr)// materials
{
    build_graph_color(topology, static_cast<int>(geometry.size()),colors);
    build_thread_by_color(colors);
}

