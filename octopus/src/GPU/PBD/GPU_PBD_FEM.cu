#include "GPU/PBD/GPU_PBD_FEM.h"

#include <set>
#include <GPU/CUMatrix.h>
#include <Manager/Debug.h>
#include <GPU/PBD/GPU_PBD_FEM_Materials.h>
#include <GPU/GPU_FEM.h>
#include <Mesh/Converter/VTK_Formater.h>
#include <Tools/Graph.h>


__device__ scalar xpbd_solve(const int nb_vert_elem, const scalar stiffness, const scalar dt, const scalar& C, const Vector3* grad_C, const scalar* inv_mass, const int* topology, Vector3* p)
{
    scalar sum_norm_grad = 0.f;
    for (int i = 0; i < nb_vert_elem; ++i) {
        sum_norm_grad += glm::dot(grad_C[i], grad_C[i]) * inv_mass[topology[i]];
    }
    //if(sum_norm_grad < 1e-12) return 0;
    const scalar alpha = 1.f / (stiffness * dt * dt);
    const scalar dt_lambda = -C / (sum_norm_grad + alpha);
    for (int i = 0; i < nb_vert_elem; ++i) {
        const int vid = topology[i];
        p[vid] += dt_lambda * inv_mass[vid] * grad_C[i];
    }
    return dt_lambda;
}

__device__ void xpbd_constraint_fem_eval(
    const Material material, const int m, const scalar lambda, const scalar mu,
    const int nb_vert_elem, const Matrix3x3& Jx_inv, const scalar& V, const Vector3* dN, const Vector3* p, const int* topology, scalar& C, Vector3* grad_C)
{

    const Matrix3x3 Jx = compute_transform(nb_vert_elem, p, topology, dN);
    const Matrix3x3 F = Jx * Jx_inv;
    Matrix3x3 P;
    scalar energy;

    eval_material(material, m, lambda, mu, F, P, energy);
    P = P * glm::transpose(Jx_inv) * V;
    C += energy * V;

    for (int i = 0; i < nb_vert_elem; ++i) {
        grad_C[i] += P * dN[i];
    }
}

__device__ void xpbd_convert_to_constraint(const int nb_vert_elem, scalar& C, Vector3* grad_C)
{
    // convert force to constraint gradient
    C = (C < 0.f) ? -C : C; // abs
    C = std::sqrt(C);
    const scalar C_inv = C < 1e-12 ? 0.f : 1.f / (2.f * C);
    for (int i = 0; i < nb_vert_elem; ++i) {
        grad_C[i] *= C_inv;
    }
}

__global__ void kernel_XPBD_V0(
    const int n, const int offset, const scalar dt, const int* eids,
    Material_Data mt,
    Vector3* p, scalar* w,
    GPU_FEM_Pameters fem, scalar* lambda)
{

    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    const int eid = eids[tid + offset];
    const int vid = eid * fem.elem_nb_vert; // first vertice id in topology
    const int qid = eid * fem.nb_quadrature;
    const int* topology = fem.topology+vid;
    for(int m = 0; m < 2; ++m) // nb materials
    {
        Vector3 grad_C[27];
        scalar C = 0.f;
        for (int j = 0; j < fem.elem_nb_vert; ++j)
            grad_C[j] = Vector3(0, 0, 0);

        for (int q = 0; q < fem.nb_quadrature; ++q) {
            Matrix3x3 JX_inv = fem.JX_inv[qid + q];
            scalar V = fem.V[qid + q];
            const Vector3* dN = fem.dN + q * fem.elem_nb_vert;
            xpbd_constraint_fem_eval(mt.material, m, mt.lambda, mt.mu, fem.elem_nb_vert, JX_inv, V, dN, p, topology, C, grad_C);
        }

        xpbd_convert_to_constraint(fem.elem_nb_vert, C, grad_C);
        //if(C < 1e-12f) continue;

        const scalar dt_lambda = xpbd_solve(fem.elem_nb_vert,(m==0) ? mt.lambda : mt.mu, dt, C, grad_C, w, topology, p);
        lambda[eid * 2 + m] = dt_lambda;
    }
}

__global__ void kernel_XPBD_primal_residual_init(
    const int n, const scalar dt, GPU_ParticleSystem_Parameters ps,
    Vector3* primal_residuals) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    primal_residuals[tid] = ps.m[tid] * (ps.p[tid] - (ps.last_p[tid] + ps.v[tid] * dt));
}


__global__ void kernel_XPBD_constraint_residuals(
    const int n, const int offset, const scalar dt, const int* eids, scalar* lambda,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    Vector3* primal_residuals, scalar* dual_residuals)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    const int eid = eids[tid + offset];
    const int vid = eid * fem.elem_nb_vert; // first vertice id in topology
    const int qid = eid * fem.nb_quadrature;
    const int* topology = fem.topology+vid;
    dual_residuals[eid] = 0;
    for(int m = 0; m < 2; ++m) // nb materials
    {
        Vector3 grad_C[32];
        scalar C = 0.f;
        for (int j = 0; j < fem.elem_nb_vert; ++j)
            grad_C[j] = Vector3(0, 0, 0);

        for (int q = 0; q < fem.nb_quadrature; ++q) { // must be possible to do in parrallel
            Matrix3x3 JX_inv = fem.JX_inv[qid + q];
            scalar V = fem.V[qid + q];
            const Vector3* dN = fem.dN + q * fem.elem_nb_vert;
            xpbd_constraint_fem_eval(mt.material, m, mt.lambda, mt.mu, fem.elem_nb_vert, JX_inv, V, dN, ps.p, topology, C, grad_C);
        }

        xpbd_convert_to_constraint(fem.elem_nb_vert, C, grad_C);
        const scalar alpha = 1.f / ((m == 0 ? mt.lambda : mt.mu) * dt * dt);
        dual_residuals[eid] += C + alpha * lambda[tid * 2 + m];
        for(int j = 0; j < fem.elem_nb_vert; ++j) {
            primal_residuals[tid] -= grad_C[j] * lambda[tid * 2 + m];
        }
    }
}



__global__ void kernel_XPBD_V1(
    const int n, const int offset, const scalar dt, const int* eids,
    Material_Data mt,
    Vector3* p, scalar* w,
    GPU_FEM_Pameters fem)
{
    if (blockIdx.x >= n) return;
    if (threadIdx.x >= fem.nb_quadrature) return;
    const int eid = eids[blockIdx.x + offset];
    const int off_topo = eid * fem.elem_nb_vert; // first vertice id in topology
    const int q_off = eid * fem.nb_quadrature;
    const int* topology = fem.topology+off_topo;
    const int& q = threadIdx.x;

    Vector3 _grad_C[27];
    extern __shared__ scalar s_C[];
    Vector3* s_grad_C = reinterpret_cast<Vector3*>(s_C + fem.elem_nb_vert);

    //__shared__ scalar s_C[27];
    //__shared__ Vector3 s_grad_C[27 * 27];
    for(int m = 0; m < 2; ++m)
    {
        scalar _C = 0;
        for (int j = 0; j < fem.elem_nb_vert; ++j)
            _grad_C[j] = Vector3(0, 0, 0);

        const Matrix3x3 JX_inv = fem.JX_inv[q_off + q];
        const scalar V = fem.V[q_off + q];
        const Vector3* dN = fem.dN + q * fem.elem_nb_vert;
        xpbd_constraint_fem_eval(mt.material, m, mt.lambda, mt.mu, fem.elem_nb_vert, JX_inv, V, dN, p, topology, _C, _grad_C);

        s_C[q] = _C;
        for(int i = 0; i < fem.elem_nb_vert; ++i) {
            s_grad_C[q * fem.elem_nb_vert + i] = _grad_C[i];
        }

        all_reduction<scalar>(threadIdx.x, fem.nb_quadrature, 0, 1,  s_C);
        all_reduction<Vector3>(threadIdx.x, fem.nb_quadrature, 0, fem.elem_nb_vert,  s_grad_C);

        const scalar stiffness = m==0 ? mt.lambda : mt.mu;
        const scalar alpha = 1.f / (stiffness * dt * dt);
        _C = s_C[0];
        _C = _C < 0.f ? -_C : _C; // abs
        _C = std::sqrt(_C);
        const scalar C_inv = _C < 1e-12 ? 0.f : 1.f / (2.f * _C);

        const int steps = (fem.elem_nb_vert-1) / fem.nb_quadrature + 1;
        for(int i = 0; i < steps; ++i)
        {
            const int id = q + fem.nb_quadrature * i;
            if(id >= fem.elem_nb_vert) continue;
            s_C[id] = glm::dot(s_grad_C[id] * C_inv, s_grad_C[id] * C_inv) * w[topology[id]];
        }

        all_reduction<scalar>(q, fem.elem_nb_vert, 0, 1,  s_C);

        const scalar sum_norm_grad = s_C[0];
        const scalar dt_lambda = -_C / (sum_norm_grad + alpha);
        for(int i = 0; i < steps; ++i)
        {
            const int id = q + fem.nb_quadrature * i;
            if(id >= fem.elem_nb_vert) continue;
            const int vid = topology[id];
            p[vid] += dt_lambda * w[vid] * C_inv * s_grad_C[id]  ;
        }
    }
}

__global__ void kernel_XPBD_phamtoms_init(
    const int n, GPU_ParticleSystem_Parameters ps,
    Vector3* f_p, scalar* f_w,
    const int* f_ids, const int* f_nb, const int* f_off
    ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (i >= n) return;
    const int nb = f_nb[i];
    const int off = f_off[i];
    const int* ids = f_ids + off;
    for(int j = 0; j < nb; ++j)
    {
        f_p[ids[j]] = ps.p[i];
        f_w[ids[j]] = ps.w[i];
    }
}

__global__ void kernel_XPBD_phamtoms_mean(
    const int n, GPU_ParticleSystem_Parameters ps,
    const Vector3* f_p,
    const int* f_ids, const int* f_nb, const int* f_off
    ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (i >= n) return;
    const int nb = f_nb[i];
    const int off = f_off[i];
    const int* ids = f_ids + off;
    Vector3 p(0,0,0);
    for(int j = 0; j < nb; ++j)
    {
        p += f_p[ids[j]];
    }
    ps.p[i] = p * (1.f / nb);
}


// MUST BE ELSEWERE
void GPU_PBD_FEM::build_graph_color(Element element, const Mesh::Topology &topology, std::vector<int>& colors) {
    Graph d_graph(element, topology, V_Dual);
    auto [nb_color, l_colors] = GraphColoration::DSAT(d_graph);

    std::cout << "NB COLORS: " << nb_color << std::endl;

    if(!use_phantom)
    {
        colors = l_colors;
        d_thread->nb_kernel = nb_color;
        return;
    }
    /// RECOLOR
    const int t_color = nb_color / 2;
    // change the coloration to match t_color and try to minimise splitting
    int total_conflict = 0;
    for(int eid = 0; eid < l_colors.size(); ++eid)
    {
        // if the color is too high
        if(l_colors[eid] < t_color) continue;

        // get for each color the number of split that it will create and with witch element
        std::vector<int> nb_split(t_color, 0); // nb split per color
        std::vector<std::set<int>> c_conflict(t_color); // elements in conflict for each cikir
        for(int eid2 : d_graph.adj[eid])
        {
            const int c = l_colors[eid2];
            if(c >= t_color) continue;
            c_conflict[c].insert(eid2);
            // the number of edges in dual graph gives
            //its not the exact number of split
            //(if there is already a conflit with another element, this may not create a new split)
            //(but this is a good approximation of the worst case)
            nb_split[c] += d_graph.edge_count(eid, eid2);
        }

        // get color with minimal split
        const auto it = std::min_element(nb_split.begin(), nb_split.end());
        const int cid = std::distance(nb_split.begin(), it);

        l_colors[eid] = cid;
        total_conflict += nb_split[cid];
    }
    colors = l_colors;
    d_thread->nb_kernel = t_color;
}

void GPU_PBD_FEM::build_phantom_particles(Element element, const Mesh::Topology &topology, std::vector<int>& colors)
{
    const int t_color = *std::max_element(colors.begin(), colors.end()) + 1;
    const int nb_verts = *std::max_element(topology.begin(), topology.end()) + 1;
    const int nb_verts_element = elem_nb_vertices(element);
    /// SPLIT
    Mesh::Topology new_topology(topology);
    int new_nb_verts = nb_verts;
    std::vector<int> f_particles;
    std::vector<int> f_offset;
    std::vector<int> f_nb;

    std::set<std::set<int>> v_conflict;
    std::vector<std::vector<int>> e_owners(nb_verts);
    std::vector<std::vector<int>> e_ref_id(nb_verts);
    // for each vertice get all its owner element
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            e_owners[topology[i + j]].push_back(i / d_fem->elem_nb_vert);
            e_ref_id[topology[i+j]].push_back(j);
        }
    }

    int f_off = 0;
    for (int i = 0; i < nb_verts; ++i)
    {
        std::vector<int> c_count(t_color,0);
        std::vector<std::vector<std::pair<int, int>>> e_conflict(t_color);
        int max_conflict = 0;
        // get all color element around the vertex
        for(int j = 0; j < e_owners[i].size(); ++j)
        {
            const int eid = e_owners[i][j];
            const int rid = e_ref_id[i][j];
            const int c = colors[eid];
            e_conflict[c].push_back({eid, rid});
            max_conflict = std::max(max_conflict, static_cast<int>(e_conflict[c].size()));
        }

        f_particles.push_back(i);
        f_nb.push_back(max_conflict);
        f_offset.push_back(f_off);
        f_off += max_conflict;

        // if there is duplicate colors around
        if(max_conflict == 1) continue;

        // groups element
        std::vector<std::vector<std::pair<int, int>>> e_groups(max_conflict);
        for(std::vector<std::pair<int, int>>& c : e_conflict)
        {
            for(int j = 0; j < c.size(); j++)
            {
                e_groups[j].push_back(c[j]);
            }
        }


        for(int j = 1; j < e_groups.size(); j++)
        {
            for( auto [eid, rid] : e_groups[j])
            {
                int e_off = eid * nb_verts_element;
                new_topology[e_off + rid] = new_nb_verts;

            }
            f_particles.push_back(new_nb_verts);
            new_nb_verts++;
        }
    }
    cb_f_topology = new Cuda_Buffer<int>(new_topology);
    cb_f_ids = new Cuda_Buffer<int>(f_particles);
    cb_f_position = new Cuda_Buffer<Vector3>(new_nb_verts);
    cb_f_inv_mass = new Cuda_Buffer<scalar>(new_nb_verts);
    cb_f_offset = new Cuda_Buffer<int>(f_offset);
    cb_f_nb = new Cuda_Buffer<int>(f_nb);
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
        d_thread->nb_threads.push_back(nb); // grid * block
        d_thread->block_size.push_back(d_fem->nb_quadrature); // nb quadrature
        d_thread->grid_size.push_back(nb); // nb constraint
    }

    cb_eid = new Cuda_Buffer(eids);
}


GPU_PBD_FEM::GPU_PBD_FEM(const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                         const scalar young, const scalar poisson, const Material material, const bool use_phantom)
        : GPU_FEM(element, geometry, topology, young, poisson, material), cb_eid(nullptr), use_phantom(use_phantom)// materials
{
    build_graph_color(element, topology, colors);
    if(use_phantom)
    {
        build_phantom_particles( element, topology, colors);
    }
    build_thread_by_color(colors);

    int s_max = 0;
    for(int k = 0; k < d_thread->nb_kernel; ++k) {
        s_max = std::max(s_max, d_thread->grid_size[k]);
    }

    cb_lambda = new Cuda_Buffer<scalar>(topology.size() / d_fem->elem_nb_vert * 2);
    cb_internia_residual = new Cuda_Buffer<Vector3>(geometry.size());
    cb_constraint_residual = new Cuda_Buffer<scalar>(topology.size() / d_fem->elem_nb_vert);

    shared_size = d_fem->elem_nb_vert * sizeof(scalar) + d_fem->elem_nb_vert * d_fem->nb_quadrature * sizeof(scalar) * 3;

    /*std::vector<scalar> c(colors.begin(), colors.end());
    std::map<Element, Mesh::Topology> topologies;
    topologies[element] = topology;
    VTK_Formater vtk;
    vtk.open("xpbd_mesh_coloration_" + element_name(element) + "_" + std::to_string(nb_color));
    vtk.save_mesh(geometry, topologies);
    vtk.start_cell_data();
    vtk.add_scalar_data(c, "colors");
    vtk.close();*/
}

void GPU_PBD_FEM::get_residuals(GPU_ParticleSystem* ps, const scalar dt, scalar& primal, scalar& dual) {
    const int nb_verts = ps->nb_particles();
    kernel_XPBD_primal_residual_init<<<nb_verts / 32 + 1, 32>>>(nb_verts, dt, ps->get_parameters(), cb_internia_residual->buffer);

    for (int j = 0; j < d_thread->nb_kernel; ++j) {
        kernel_XPBD_constraint_residuals<<<d_thread->nb_threads[j] / 32 + 1, 32>>>(
            d_thread->nb_threads[j], d_thread->offsets[j], dt,
            cb_eid->buffer, cb_lambda->buffer, *d_material,
            ps->get_parameters(), get_fem_parameters(), cb_internia_residual->buffer, cb_constraint_residual->buffer
            );
    }

    dual = 0;
    std::vector<scalar> data;
    cb_constraint_residual->get_data(data);
    for(int i = 0; i < data.size(); ++i) {
        dual += data[i] * data[i];
    }
    dual/= data.size();

    primal = 0;
    std::vector<Vector3> vdata;
    cb_internia_residual->get_data(vdata);
    for(int i = 0; i < vdata.size(); ++i) {
        primal += glm::length2(vdata[i]);
    }
    primal/= vdata.size();
}

void GPU_PBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {

    GPU_FEM_Pameters fem_pameters = get_fem_parameters();
    GPU_ParticleSystem_Parameters ps_parm = ps->get_parameters();
    Vector3* p;
    scalar* w;
    if(use_phantom)
    {
        p = cb_f_position->buffer;
        w = cb_f_inv_mass->buffer;
        fem_pameters.topology = cb_f_topology->buffer;

        const int n = ps->nb_particles();
        kernel_XPBD_phamtoms_init<<<(n+31)/32, 32>>>(
        n, ps_parm,
        p, w,
        cb_f_ids->buffer, cb_f_nb->buffer, cb_f_offset->buffer);
    }
    else
    {
        p = ps_parm.p;
        w = ps_parm.w;
    }




    for (int j = 0; j < d_thread->nb_kernel; ++j) {
        /*kernel_XPBD_V0<<<d_thread->nb_threads[j] / 32 + 1, 32>>>(
            d_thread->nb_threads[j], d_thread->offsets[j], dt,
            cb_eid->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), cb_lambda->buffer);/**/

        if(d_fem->nb_element == 4)
        {
            kernel_XPBD_V0<<<d_thread->nb_threads[j] / 32 + 1, 32>>>(
            d_thread->nb_threads[j], d_thread->offsets[j], dt,
            cb_eid->buffer, *d_material, p, w, fem_pameters, cb_lambda->buffer);
        }
        else
        {
            kernel_XPBD_V1<<<d_thread->grid_size[j], d_thread->block_size[j], shared_size>>>(
               d_thread->nb_threads[j], d_thread->offsets[j], dt,
               cb_eid->buffer, *d_material, p, w, fem_pameters
           );
        }/**/
    }

    if(use_phantom)
    {
        int n = ps->nb_particles();
        kernel_XPBD_phamtoms_mean<<<(n+31)/32, 32>>>(
            n, ps->get_parameters(),
            p,
            cb_f_ids->buffer, cb_f_nb->buffer, cb_f_offset->buffer);
    }
    /*scalar p, d;
    get_residuals(ps, dt,p, d);
    DebugUI::Begin("XPBD  residual");
    DebugUI::Value("p value", p);
    DebugUI::Plot("p plot",  p);
    DebugUI::Range("p range", p);
    DebugUI::Value("d value", d);
    DebugUI::Plot("d plot",  d);
    DebugUI::Range("d range", d);
    DebugUI::End();*/
}
