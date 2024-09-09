#include "GPU/GPU_VBD.h"
#include <set>


GPU_VBD_FEM::GPU_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const scalar &young, const scalar &poisson) {
    const int nb_vertices = static_cast<int>(geometry.size());
    shape = get_fem_shape(element);
    shape->build();

    lambda = young * poisson / ((1.f + poisson) * (1.f - 2.f * poisson));
    mu = young / (2.f * (1.f + poisson));

    elem_nb_vert = elem_nb_vertices(element);
    nb_quadrature = static_cast<int>(shape->weights.size());
    nb_element = static_cast<int>(topology.size()) / elem_nb_vert;
    std::vector<std::vector<int>> e_neighbors;
    std::vector<std::vector<int>> e_ref_id;
    build_graph_color(topology, nb_vertices, colors,e_neighbors,e_ref_id);
    sort_by_color(nb_vertices, e_neighbors, e_ref_id);
    build_fem_const(geometry, topology);
    cb_topology = new Cuda_Buffer(topology);
}

void GPU_VBD_FEM::sort_by_color(int nb_vertices, const std::vector<std::vector<int>>& e_neighbors, const std::vector<std::vector<int>>& e_ref_id) {
    // sort by color
    std::vector<int> ref_id;
    std::vector<int> neighbors;
    std::vector<int> nb_neighbors;
    std::vector<int> neighbors_offset;
    // sort neighbors
    for(int c = 0; c < nb_color; ++c) {
        int n_max = 1;
        int nb_thread = 0;
        int n = static_cast<int>(nb_neighbors.size());
        for(int i = 0; i < nb_vertices; ++i) {
            if(c != colors[i]) continue;
            neighbors_offset.push_back(static_cast<int>(neighbors.size()));
            neighbors.insert(neighbors.end(), e_neighbors[i].begin(), e_neighbors[i].end());
            ref_id.insert(ref_id.end(), e_ref_id[i].begin(), e_ref_id[i].end());
            nb_neighbors.push_back(static_cast<int>(e_neighbors[i].size()));

            n_max = std::max(n_max, nb_neighbors.back());
            nb_thread ++;
        }
        c_block_size.push_back(n_max * nb_quadrature);
        c_nb_threads.push_back(nb_thread * n_max * nb_quadrature);
        c_offsets.push_back(n);
    }
    cb_nb_neighbors = new Cuda_Buffer(nb_neighbors);
    cb_neighbors = new Cuda_Buffer(neighbors);
    cb_neighbors_offset = new Cuda_Buffer(neighbors_offset);
    cb_ref_vid = new Cuda_Buffer(ref_id);
}

void GPU_VBD_FEM::build_fem_const(const Mesh::Geometry &geometry, const Mesh::Topology& topology) {
    std::vector<Vector3> dN;
    std::vector<Matrix3x3> JX_inv(nb_quadrature * nb_element);
    std::vector<scalar> V(nb_quadrature * nb_element);
    for (int i = 0; i < nb_quadrature; i++)
        dN.insert(dN.end(), shape->dN[i].begin(), shape->dN[i].end());

    for (int i = 0; i < nb_element; i++) {
        const int id = i * elem_nb_vert;
        const int eid = i * nb_quadrature;
        for (int j = 0; j < nb_quadrature; ++j) {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < elem_nb_vert; ++k) {
                J += glm::outerProduct(geometry[topology[id + k]], dN[j * elem_nb_vert + k]);
            }
            V[eid + j] = abs(glm::determinant(J)) * shape->weights[j];
            JX_inv[eid + j] = glm::inverse(J);
        }
    }

    cb_dN = new Cuda_Buffer(dN);
    cb_V = new Cuda_Buffer(V);
    cb_JX_inv = new Cuda_Buffer(JX_inv);
}


void GPU_VBD_FEM::build_graph_color(const Mesh::Topology &topology, int nb_vertices,
    std::vector<int> &colors, std::vector<std::vector<int>>& e_neighbors, std::vector<std::vector<int>>& e_ref_id) {
    std::vector<std::set<int> > neighbors(nb_vertices);
    e_neighbors.resize(nb_vertices);
    e_ref_id.resize(nb_vertices);
    // for each vertice get all its neighboors
    for (int i = 0; i < topology.size(); i += elem_nb_vert) {
        for (int j = 0; j < elem_nb_vert; ++j) {
            e_neighbors[topology[i + j]].push_back(i / elem_nb_vert);
            e_ref_id[topology[i+j]].push_back(j);
            // all vertices inside an element are linked
            for (int k = 0; k < elem_nb_vert; ++k) {
                if (k == j) continue;
                neighbors[topology[i + j]].insert(topology[i + k]);
            }
        }
    }
    int max_neighbors = 0;
    nb_color = 1;
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
                nb_color = std::max(nb_color, c);
                colors[i] = c;
                break;
            }
        }
        std::fill(available.begin(), available.end(), true);
    }
    nb_color++;
    std::cout << "NB color: " << nb_color << "  NB neighbors : " << max_neighbors << std::endl;
}
