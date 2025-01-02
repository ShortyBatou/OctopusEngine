#include "GPU/VBD/GPU_MG_VBD_FEM.h"
#include <GPU/GPU_FEM_Material.h>
#include <random>
#include <numeric>
#include <set>
#include <Dynamic/VBD/MG_VertexBlockDescent.h>

GPU_MG_VBD_FEM::GPU_MG_VBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry,
                               const Material& material, const scalar& young, const scalar& poisson,
                               const scalar& damping, const scalar& linear, const int& nb_iteration) :
    GPU_VBD_FEM(element, topology, geometry, material, young, poisson, damping)
{
    assert(element == Tetra10 || element == Hexa27);

    int it_linear = static_cast<int>(static_cast<scalar>(nb_iteration) * linear);
    int it_quad = nb_iteration - it_linear;
    nb_iterations = std::vector<int>({it_quad, it_linear});
    it_count = 0;
    level = 1;

    const int nb_vertices = static_cast<int>(geometry.size());
    const int nb_vert_elem = elem_nb_vertices(element);
    const int nb_elem = static_cast<int>(topology.size()) / nb_vert_elem;

    // get linear topology (could be nice to have that in a global function)
    const Element lin_elem = element == Tetra10 ? Tetra : Hexa;
    const int lin_nb_vert_elem = elem_nb_vertices(lin_elem);
    std::vector<int> lin_topo(nb_elem * lin_nb_vert_elem);
    for (int i = 0; i < nb_elem; i++)
        for (int j = 0; j < lin_nb_vert_elem; ++j)
            lin_topo[i * lin_nb_vert_elem + j] = lin_topo[i * nb_vert_elem + j];

    // fem data of quadratic and linear element
    l_fems.push_back(d_fem);
    l_fems.push_back(GPU_FEM::build_fem_const(lin_elem, geometry, topology));

    // thread data
    l_threads.push_back(d_thread);
    l_threads.push_back(new Thread_Data());

    // prepare FEM to build new thread data for linear fem
    d_thread = l_threads.back();
    d_fem = l_fems.back();

    std::vector<std::vector<int>> e_neighbors;
    std::vector<std::vector<int>> e_ref_id;
    build_graph_color(topology, nb_vertices, colors, e_neighbors, e_ref_id);
    sort_by_color(nb_vertices, e_neighbors, e_ref_id);

    // build interpolations
    if(element == Tetra10)
    {
        P1_to_P2 interpolation(topology);
        GPU_MG_Interpolation* i_mid_edge = new GPU_MG_Interpolation();
        i_mid_edge->cb_ids = new Cuda_Buffer<int>(interpolation.ids);
        i_mid_edge->cb_primitives = new Cuda_Buffer<int>(interpolation.edges);
        i_mid_edge->nb_vert_primitives = 2;
        i_mid_edge->weight = 0.5;
        interpolations.push_back(i_mid_edge);
    }
    else
    {
        Q1_to_Q2 interpolation(topology);
        GPU_MG_Interpolation* i_mid_edge = new GPU_MG_Interpolation();
        i_mid_edge->cb_ids = new Cuda_Buffer<int>(interpolation.ids_edges);
        i_mid_edge->cb_primitives = new Cuda_Buffer<int>(interpolation.edges);
        i_mid_edge->nb_vert_primitives = 2;
        i_mid_edge->weight = 0.5;
        interpolations.push_back(i_mid_edge);

        GPU_MG_Interpolation* i_mid_face = new GPU_MG_Interpolation();
        i_mid_face->cb_ids = new Cuda_Buffer<int>(interpolation.ids_faces);
        i_mid_face->cb_primitives = new Cuda_Buffer<int>(interpolation.faces);
        i_mid_face->nb_vert_primitives = 4;
        i_mid_face->weight = 0.25;
        interpolations.push_back(i_mid_face);

        GPU_MG_Interpolation* i_mid_volume = new GPU_MG_Interpolation();
        i_mid_volume->cb_ids = new Cuda_Buffer<int>(interpolation.ids_volumes);
        i_mid_volume->cb_primitives = new Cuda_Buffer<int>(interpolation.faces);
        i_mid_volume->nb_vert_primitives = 8;
        i_mid_volume->weight = 0.125;
        interpolations.push_back(i_mid_volume);
    }


}
