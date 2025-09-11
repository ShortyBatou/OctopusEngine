#include "GPU/PBD/GPU_PBD_FEM.h"

#include <set>
#include <GPU/CUMatrix.h>
#include <Manager/Debug.h>
#include <GPU/PBD/GPU_PBD_FEM_Materials.h>
#include <GPU/GPU_FEM.h>


__device__ void xpbd_solve(const int nb_vert_elem, const scalar stiffness, const scalar dt, const scalar& C, const Vector3* grad_C, const scalar* inv_mass, const int* topology, Vector3* p, const int* mask)
{
    scalar sum_norm_grad = 0.f;
    for (int i = 0; i < nb_vert_elem; ++i) {
        sum_norm_grad += glm::dot(grad_C[i], grad_C[i]) * inv_mass[topology[i]];
    }
    if(sum_norm_grad < 1e-12) return;
    const scalar alpha = 1.f / (stiffness * dt * dt);
    const scalar dt_lambda = -C / (sum_norm_grad + alpha);
    for (int i = 0; i < nb_vert_elem; ++i) {
        const int vid = topology[i];
        if(mask[vid] == 1) p[vid] += dt_lambda * inv_mass[vid] * grad_C[i];
    }
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
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem)
{

    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread it
    if (tid >= n) return;
    const int eid = eids[tid + offset];
    const int vid = eid * fem.elem_nb_vert; // first vertice id in topology
    const int qid = eid * fem.nb_quadrature;
    const int* topology = fem.topology+vid;
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
        if(C < 1e-12f) continue;

        xpbd_solve(fem.elem_nb_vert,(m==0) ? mt.lambda : mt.mu, dt, C, grad_C, ps.w, topology, ps.p, ps.mask);
    }
}

__global__ void kernel_XPBD_V1(
    const int n, const int offset, const scalar dt, const int* eids,
    int m, Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem, scalar* C, Vector3* grad_C)
{
    if (blockIdx.x >= n) return;
    if (threadIdx.x >= fem.nb_quadrature) return;
    const int eid = eids[blockIdx.x + offset];
    const int vid = eid * fem.elem_nb_vert; // first vertice id in topology
    const int qid = eid * fem.nb_quadrature;
    const int* topology = fem.topology+vid;
    const int& q = threadIdx.x;
    //printf("%d %d\n", blockIdx.x, q);

    scalar _C = threadIdx.x + 1;
    Vector3 _grad_C[32];
    for (int j = 0; j < fem.elem_nb_vert; ++j)
        _grad_C[j] = Vector3(0, 0, 0);

    const Matrix3x3 JX_inv = fem.JX_inv[qid + q];
    const scalar V = fem.V[qid + q];
    const Vector3* dN = fem.dN + q * fem.elem_nb_vert;
    xpbd_constraint_fem_eval(mt.material, m, mt.lambda, mt.mu, fem.elem_nb_vert, JX_inv, V, dN, ps.p, topology, _C, _grad_C);

    __shared__ scalar s_C[32];
    __shared__ Vector3 s_grad_C[512];
    s_C[q] = _C;
    for(int i = 0; i < fem.elem_nb_vert; ++i) {
        s_grad_C[q * fem.elem_nb_vert + i] = _grad_C[i];
    }

    all_reduction<scalar>(threadIdx.x, fem.nb_quadrature, 0, 1,  s_C);
    all_reduction<Vector3>(threadIdx.x, fem.nb_quadrature, 0, fem.elem_nb_vert,  s_grad_C);

    if(threadIdx.x == 0) {
        _C = s_C[0];
        for(int i = 0; i < fem.elem_nb_vert; ++i)
            _grad_C[i] = s_grad_C[i];

        xpbd_convert_to_constraint(fem.elem_nb_vert, _C, _grad_C);

        C[blockIdx.x] = _C;
        for (int i = 0; i < fem.elem_nb_vert; ++i) {
            grad_C[blockIdx.x * fem.elem_nb_vert + i] = _grad_C[i];
        }
    }
}

__global__ void kernel_XPBD_Solve_V1(
    const int n, const int offset, const scalar dt, const int* eids, int m, Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem, scalar* C, Vector3* grad_C)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    //printf("%d\n", tid);
    const int eid = eids[tid + offset];
    const int vid = eid * fem.elem_nb_vert; // first vertice id in topology
    const int* topology = fem.topology+vid;
    if(C[tid] < 1e-6f) return;
    xpbd_solve(fem.elem_nb_vert,(m==0)?mt.lambda:mt.mu, dt, C[tid], grad_C + tid * fem.elem_nb_vert, ps.w, topology, ps.p, ps.mask);
}


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
        d_thread->nb_threads.push_back(nb); // grid * block
        d_thread->block_size.push_back(d_fem->nb_quadrature); // nb quadrature
        d_thread->grid_size.push_back(nb); // nb constraint
    }

    cb_eid = new Cuda_Buffer(eids);
}


GPU_PBD_FEM::GPU_PBD_FEM(const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                         const scalar young, const scalar poisson, const Material material)
        : GPU_FEM(element, geometry, topology, young, poisson, material), cb_eid(nullptr)// materials
{
    build_graph_color(topology, static_cast<int>(geometry.size()),colors);
    build_thread_by_color(colors);

    int s_max = 0;
    for(int k = 0; k < d_thread->nb_kernel; ++k) {
        s_max = std::max(s_max, d_thread->grid_size[k]);
    }
    cb_C = new Cuda_Buffer<scalar>(s_max);
    cb_grad_C = new Cuda_Buffer<Vector3>(s_max * elem_nb_vertices(element));
}

scalar GPU_PBD_FEM::get_inertia_residual() {
 return 0;
}

scalar GPU_PBD_FEM::get_constraint_residual() {
    return 0;
}

void GPU_PBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    for (int j = 0; j < d_thread->nb_kernel; ++j) {
        kernel_XPBD_V0<<<d_thread->nb_threads[j] / 32 + 1, 32>>>(
            d_thread->nb_threads[j], d_thread->offsets[j], dt,
            cb_eid->buffer, *d_material, ps->get_parameters(), get_fem_parameters());

        /*for(int m = 0; m < 2; ++m) {
            kernel_XPBD_V1<<<d_thread->grid_size[j], d_thread->block_size[j]>>>(
               d_thread->nb_threads[j], d_thread->offsets[j], dt,
               cb_eid->buffer, m, *d_material, ps->get_parameters(), get_fem_parameters(),
               cb_C->buffer, cb_grad_C->buffer
           );
            kernel_XPBD_Solve_V1<<< (d_thread->grid_size[j] + 31) / 32, 32>>>(
               d_thread->grid_size[j], d_thread->offsets[j], dt,
               cb_eid->buffer, m, *d_material, ps->get_parameters(), get_fem_parameters(),
               cb_C->buffer, cb_grad_C->buffer
           );
        }*/

    }
}
