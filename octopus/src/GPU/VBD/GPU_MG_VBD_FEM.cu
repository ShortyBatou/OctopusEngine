#include "GPU/VBD/GPU_MG_VBD_FEM.h"
#include <Dynamic/VBD/MG_VertexBlockDescent.h>
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
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


GPU_MG_VBD_FEM::GPU_MG_VBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry,
                               const Material& material, const scalar& young, const scalar& poisson,
                               const scalar& damping, const scalar& linear, const int& nb_iteration,
                               const scalar& density, const Mass_Distribution& mass_distrib,
                               GPU_ParticleSystem* ps) :
    GPU_VBD_FEM(element, topology, geometry, material, young, poisson, damping)
{
    assert(element == Tetra10 || element == Hexa27);

    int it_linear = static_cast<int>(static_cast<scalar>(nb_iteration) * linear);
    int it_quad = nb_iteration - it_linear;
    nb_iterations = std::vector<int>({it_quad, it_linear});
    it_count = 0;
    level = 1;

    const int nb_vert_elem = elem_nb_vertices(element);
    const int nb_elem = static_cast<int>(topology.size()) / nb_vert_elem;

    // get linear topology (could be nice to have that in a global function)
    const Element lin_elem = element == Tetra10 ? Tetra : Hexa;
    const int lin_nb_vert_elem = elem_nb_vertices(lin_elem);
    std::set<int> vids;
    std::vector<int> lin_topo(nb_elem * lin_nb_vert_elem);
    for (int i = 0; i < nb_elem; i++)
    {
        for (int j = 0; j < lin_nb_vert_elem; ++j)
        {
            int vid = topology[i * nb_vert_elem + j];
            lin_topo[i * lin_nb_vert_elem + j] = vid;
            vids.insert(vid);
        }

    }
    int lin_nb_vertices = static_cast<int>(vids.size());


    // fem data of quadratic and linear element
    l_fems.push_back(d_fem);
    l_fems.push_back(GPU_FEM::build_fem_const(lin_elem, geometry, lin_topo));

    // thread data
    l_threads.push_back(d_thread);
    l_threads.push_back(new Thread_Data());

    l_owners.push_back(d_owners);
    l_owners.push_back(new GPU_Owners_Data());
    // prepare FEM to build new thread data for linear fem
    d_thread = l_threads.back();
    d_fem = l_fems.back();
    d_owners = l_owners.back();

    std::vector<std::vector<int>> e_neighbors;
    std::vector<std::vector<int>> e_ref_id;
    std::vector<int> colors;
    build_graph_color(lin_topo, lin_nb_vertices, colors, e_neighbors, e_ref_id);
    sort_by_color(lin_nb_vertices, colors, e_neighbors, e_ref_id);

    masses.push_back(ps->_data->_cb_mass);
    std::vector<scalar> lin_masses = compute_fem_mass(lin_elem, geometry, lin_topo, density, mass_distrib);
    masses.push_back(new Cuda_Buffer<scalar>(lin_masses));

    // build interpolations
    if(element == Tetra10)
    {
        P1_to_P2 inter(topology);
        auto* i_mid_edge = new GPU_MG_Interpolation(2,0.5, inter.ids, inter.edges);
        interpolations.push_back(i_mid_edge);
    }
    else
    {
        Q1_to_Q2 inter(topology);
        auto* i_mid_edge = new GPU_MG_Interpolation(2,0.5, inter.ids_edges, inter.edges);
        interpolations.push_back(i_mid_edge);

        auto* i_mid_face = new GPU_MG_Interpolation(4,0.25,inter.ids_faces, inter.faces);
        interpolations.push_back(i_mid_face);

        auto* i_mid_volume = new GPU_MG_Interpolation(8,0.125,inter.ids_volumes, inter.volume);
        interpolations.push_back(i_mid_volume);
    }
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

    GPU_VBD_FEM::step(ps, dt);

    if (level == 1)
    {
        for(int i = 0; i < interpolations.size(); ++i)
        {
            const auto inter_param = get_interpolation_parameters(i);
            kernel_prolongation<<<(inter_param.nb_ids+31)/32,32>>>(inter_param.nb_ids, ps_param, inter_param);
        }
    }
}
