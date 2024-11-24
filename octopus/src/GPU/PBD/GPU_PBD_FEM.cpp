#include "GPU/PBD/GPU_PBD_FEM.h"
#include <set>

// MUST BE ELSEWERE
int GPU_PBD_FEM::build_graph_color(const Mesh::Topology &topology, const int nb_vert, std::vector<int>& colors) {
    nb_color = 1;
    std::vector<std::set<int> > owners(nb_vert);
    // for each vertice get elements that own this vertice
    for (int i = 0; i < topology.size(); i += elem_nb_vert) {
        for (int j = 0; j < elem_nb_vert; ++j) {
            owners[topology[i + j]].insert(i / elem_nb_vert);
        }
    }

    colors.resize(topology.size() / elem_nb_vert, -1);
    std::vector<int> available(64, true);
    for (int i = 0; i < topology.size(); i += elem_nb_vert) {
        // for all vertices, check the neighbor elements colors
        for (int j = 0; j < elem_nb_vert; ++j) {
            for (int n: owners[topology[i + j]]) {
                if (colors[n] != -1) available[colors[n]] = false;
            }
        }
        for (int c = 0; c < available.size(); ++c) {
            if (available[c]) {
                nb_color = std::max(nb_color, c);
                colors[i / elem_nb_vert] = c;
                break;
            }
        }
        std::fill(available.begin(), available.end(), true);
    }
    std::cout << "NB color: " << nb_color + 1 << std::endl;
    return nb_color + 1;
}

std::vector<int> GPU_PBD_FEM::build_topology_by_color(const std::vector<int>& colors, const std::vector<int> &topology) {
    // sort element by color and get color group sizes
    c_offsets.resize(nb_color);
    std::vector<int> sorted_topology;
    int count = 0;
    for (int c = 0; c < nb_color; ++c) {
        c_offsets[c] = count;
        for (int i = 0; i < nb_element; ++i) {
            if (colors[i] != c) continue;
            const int id = i * elem_nb_vert;
            sorted_topology.insert(sorted_topology.end(), topology.begin() + id, topology.begin() + id + elem_nb_vert);
            count += elem_nb_vert;
        }
    }
    const int s_off = static_cast<int>(c_offsets.size());
    const int s_topo = static_cast<int>(sorted_topology.size());
    // build constant value for FEM simulation and init buffers
    for (int i = 0; i < s_off; ++i) {
        const int nb = (i < s_off - 1 ? c_offsets[i + 1] : s_topo) - c_offsets[i];
        c_nb_elem.push_back(nb / elem_nb_vert);
    }

    cb_topology = new Cuda_Buffer(sorted_topology);
    return sorted_topology;
}


GPU_PBD_FEM::GPU_PBD_FEM(const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                         const scalar young, const scalar poisson, Material material) : GPU_FEM(element, geometry, topology, young, poisson, material)// materials
{
    GPU_PBD_FEM::build_fem_const(geometry, topology);
}

void GPU_PBD_FEM::build_fem_const(const Mesh::Geometry& geometry, const Mesh::Topology& topology)
{
    nb_color = build_graph_color(topology, static_cast<int>(geometry.size()),colors);
    const Mesh::Topology sorted_topology = build_topology_by_color(colors, topology);
    GPU_FEM::build_fem_const(geometry, sorted_topology);
}