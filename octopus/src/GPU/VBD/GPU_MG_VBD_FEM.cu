#include "GPU/VBD/GPU_MG_VBD_FEM.h"
#include <Dynamic/VBD/MG_VertexBlockDescent.h>
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <random>
#include <numeric>
#include <Dynamic/FEM/FEM_Generic.h>
#include <GPU/CUMatrix.h>


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

__global__ void kernel_interpolation(const int n, GPU_ParticleSystem_Parameters ps, GPU_Adjacence_Parameters adjacence) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int vid = adjacence.ids[tid];
    const int nb = adjacence.nb[tid];
    const int off = adjacence.offset[tid];
    const int* adj = adjacence.adj + off;
    const scalar* vals = adjacence.values + off;

    Vector3 dt_p = Vector3(0);
    for(int i = 0; i < nb; ++i)
        dt_p += (ps.p[adj[i]] - ps.last_p[adj[i]]) * vals[i]; // place holder

    ps.p[vid] += dt_p; // place holder
}

__global__ void kernel_restriction_intertia(
    const int n, const scalar dt, const Vector3 g,
    GPU_ParticleSystem_Parameters ps, GPU_MG_Interpolation_Parameters inter,
    Vector3* y) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int* primitive = inter.primitives + tid * inter.nb_vert_primitives;
    Vector3 v = Vector3(0);
    const int vid = inter.ids[tid];
    for(int i = 0; i < inter.nb_vert_primitives; ++i)
    {
        const int p_vid = primitive[i];
        v += ps.v[p_vid] * ps.m[p_vid] / ps.m[vid];
    }
    v *= 1.f / inter.nb_vert_primitives;

    const Vector3 a_ext = g + ps.f[vid] * ps.w[vid];
    y[vid] = ps.last_p[vid] + (ps.v[vid] + a_ext * dt) * dt;
}

__global__ void kernel_intertia(
        const scalar dt, const Vector3 g,
        GPU_ParticleSystem_Parameters ps,
        Vector3* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    const Vector3 a_ext = g + ps.f[i] * ps.w[i];
    y[i] = ps.last_p[i] + (ps.v[i] + a_ext * dt) * dt;
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


    interias.push_back(new Cuda_Buffer(geometry));
    interias.push_back(new Cuda_Buffer(geometry));
    y = interias[level];
    const int nb_vert_elem = elem_nb_vertices(element);
    const int nb_elem = static_cast<int>(topology.size()) / nb_vert_elem;

    // get linear topology (could be nice to have that in a global function)
    const Element lin_elem = get_linear_element(element);
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

    // create linear data (modify d_thread, d_owners and d_block <== this one is useless for now)
    std::vector<std::vector<int>> e_owners;
    std::vector<std::vector<int>> e_ref_id;
    build_owner_data(vids.size(), lin_topo, e_owners, e_ref_id);
    Coloration coloration = build_graph_color(lin_elem, lin_topo); // get coloration
    create_buffers(lin_elem, lin_topo, coloration, e_owners, e_ref_id);

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

    FEM_Shape* lin_shape = get_fem_shape(lin_elem);
    FEM_Shape* shape = get_fem_shape(element);
    std::vector<scalar> masses = compute_fem_mass(element, geometry, topology, density, mass_distrib);


    gpu_adjacence = compute_interpolation_adj_matrix(lin_shape, shape, topology, geometry, masses, density);

    delete shape;
    delete lin_shape;
}

 GPU_Adjacence* GPU_MG_VBD_FEM::compute_interpolation_adj_matrix(
    const FEM_Shape* from_shape, const FEM_Shape* target_shape,
    const Mesh::Topology& topology, const Mesh::Geometry& geometry,
    const std::vector<scalar>& masses, const scalar density) {
    int nb_vert_elem = target_shape->nb;
    FEM_Shape* q_shape = get_fem_shape(Tetra20);
    std::vector<scalar> quad_coord = q_shape->get_quadrature_coordinates();
    std::vector<Vector3> verts(target_shape->nb);

    // sparse matrix
    std::map<std::pair<int, int>, scalar> proj;

    // compute the projection matrix with Gauss quadratures
    for(int q = 0; q < q_shape->nb_quadratures(); ++q) {
        scalar w = q_shape->weights[q];
        scalar x = quad_coord[q * 3], y = quad_coord[q * 3 + 1], z = quad_coord[q * 3 + 2];
        std::vector<scalar> shape_func = target_shape->build_shape(x,y,z);
        std::vector<scalar> lin_shape_func = from_shape->build_shape(x,y,z);
        std::vector<Vector3> dN = target_shape->build_shape_derivatives(x, y, z);

        // get the projection for each element
        for (int i = 0; i < topology.size(); i += nb_vert_elem)
        {
            std::vector<int> e_topo(topology.begin() + i, topology.begin() + i + nb_vert_elem);
            for(int j = 0; j < target_shape->nb; j++) verts[j] = geometry[e_topo[j]];
            const scalar v = glm::determinant(FEM_Generic::get_jacobian(verts, dN));
            const scalar volume = abs(v) * w;

            // compute each value in projection matrix
            for(int k = 0; k < target_shape->nb; ++k) {
                int vid = e_topo[k];
                for(int l = 0; l < from_shape->nb; ++l) {
                    std::pair<int, int> pair(vid, e_topo[l]);
                    scalar p = shape_func[k] * lin_shape_func[l] * volume  / (masses[vid] / density);
                    if(proj.find(pair) != proj.end()) proj[pair] += p;
                    else proj[pair] = p;
                }
            }
        }
    }

    std::vector<int> nb, off, ids, adj;
    std::vector<scalar> values;
    int current_id = proj.begin()->first.first;
    int offset = 0, count = 0;
    for(auto [edge, value] : proj) {
        if(current_id != edge.first) {
            nb.push_back(count);
            off.push_back(offset);
            ids.push_back(current_id);
            current_id = edge.first;
            offset += count;
            count = 0;
        }
        if(abs(value) < 1e-3) continue;

        adj.push_back(edge.second);
        values.push_back(value);
        count++;
    }

    GPU_Adjacence* gpu_adj = new GPU_Adjacence();
    gpu_adj->cb_nb = new Cuda_Buffer<int>(nb);
    gpu_adj->cb_offset = new Cuda_Buffer<int>(off);
    gpu_adj->cb_ids = new Cuda_Buffer<int>(ids);
    gpu_adj->cb_adj = new Cuda_Buffer<int>(adj);
    gpu_adj->cb_values = new Cuda_Buffer<scalar>(values);
    return gpu_adj;
}


void GPU_MG_VBD_FEM::compute_intertia(GPU_ParticleSystem* ps, const scalar dt) const {
    kernel_intertia<<<(ps->nb_particles() + 31)/32, 32>>>(dt,Dynamic::gravity(),
        ps->get_parameters(), y->buffer);
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
            //int nb_thread = gpu_adjacence->cb_nb->nb;
            //kernel_interpolation<<<nb_thread / 32 + 1, 32>>>(nb_thread, ps_param, get_adj_parameters());
        }
    }

    ps->_data->_cb_mass = masses[level];
    d_thread = l_threads[level];
    d_fem = l_fems[level];
    d_owners = l_owners[level];
    y = interias[level];

    GPU_VBD_FEM::step(ps, dt);

    if (level == 1)
    {
        for(int i = 0; i < interpolations.size(); ++i)
        {
            const auto inter_param = get_interpolation_parameters(i);
            kernel_prolongation<<<(inter_param.nb_ids+31)/32,32>>>(inter_param.nb_ids, ps_param, inter_param);
            //int nb_thread = gpu_adjacence->cb_nb->nb;
            //kernel_interpolation<<<nb_thread / 32 + 1, 32>>>(nb_thread, ps_param, get_adj_parameters());
        }
    }
}
