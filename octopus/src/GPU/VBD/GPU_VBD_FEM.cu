#include "GPU/CUMatrix.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include "GPU/VBD/GPU_VBD_FEM.h"

#include <random>
#include <set>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/GPU_ParticleSystem.h>

__global__ void kernel_vbd_solve(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
    const int size_of_block = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    if(ps.mask[vid] == 0) return;

    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;
    //if(threadIdx.x == 0) printf("[%d][%d/%d] cid=%d, nb=%d, offset=%d, vid=%d, eid=%d, qid=%d, rid=%d, qe_off=%d, qv_off=%d \n",gid,tid+1,size_of_block,cid, nb_owners[cid],offset,vid, eid, qid, r_vid, qe_off, qv_off);
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[6];

    for (int i = 0; i < fem.elem_nb_vert; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], fem.dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * fem.JX_inv[qe_off];
    const Vector3 dF_dx = glm::transpose(fem.JX_inv[qe_off]) * fem.dN[qv_off + r_vid];

    // Compute force at vertex i
    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    Vector3 fi = -P * dF_dx * fem.V[qe_off];

    // Compute hessian
    eval_hessian(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    Matrix3x3 H = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2W_dF2);

    // shared variable : f, H
    __shared__ scalar s_f_H[2592]; // size = block_size * 12 * sizeof(float)
    int k = 0;
    for(int j = 0; j < 3; ++j) {
        s_f_H[tid * 9 + j] = fi[j];
        for(int i = j; i < 3; ++i) {
            s_f_H[tid * 9 + 3 + k] = H[i][j];
            ++k;
        }
    }

    __syncthreads();
    int t = size_of_block;
    int i,b;
    for(i=t/2, b=(t+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < 9; ++j) {
                s_f_H[tid*9+j] += s_f_H[(tid+b)*9+j];
            }
            __syncthreads();
        }
        i = (b>i) ? b : i;
    }

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[6];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[7];  H[2][2] = s_f_H[8];
        // symmetry
        H[0][1] = H[1][0];
        H[1][2] = H[2][1];
        H[0][2] = H[2][0];


        // damping (velocity)
        fi -= damping / dt * H * (ps.p[vid] - ps.last_p[vid]);
        H  += damping / dt * H;

        // intertia (accellearation)
        scalar mh2 = ps.m[vid] / (dt*dt);
        fi -= mh2 * (ps.p[vid] - y[vid]);
        H[0][0] += mh2; H[1][1] += mh2; H[2][2] += mh2;
        //scalar detH = glm::determinant(s_H);
        scalar detH = abs(glm::determinant(H));
        Vector3 dx = detH > 1e-6f ? glm::inverse(H) * fi : Vector3(0.f);
        ps.p[vid] += dx;
    }
}


__global__ void kernel_vbd_compute_residual(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    Vector3* r,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
    const int size_of_block = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    if(ps.mask[vid] == 0) return;

    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;
    //if(threadIdx.x == 0) printf("[%d][%d/%d] cid=%d, nb=%d, offset=%d, vid=%d, eid=%d, qid=%d, rid=%d, qe_off=%d, qv_off=%d \n",gid,tid+1,size_of_block,cid, nb_owners[cid],offset,vid, eid, qid, r_vid, qe_off, qv_off);
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[6];

    for (int i = 0; i < fem.elem_nb_vert; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], fem.dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * fem.JX_inv[qe_off];
    const Vector3 dF_dx = glm::transpose(fem.JX_inv[qe_off]) * fem.dN[qv_off + r_vid];

    // Compute force at vertex i
    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    Vector3 fi = -P * dF_dx * fem.V[qe_off];

    // Compute hessian
    eval_hessian(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    Matrix3x3 H = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2W_dF2);

    // shared variable : f, H
    __shared__ scalar s_f_H[2592]; // size = block_size * 12 * sizeof(float)
    int k = 0;
    for(int j = 0; j < 3; ++j) {
        s_f_H[tid * 9 + j] = fi[j];
        for(int i = j; i < 3; ++i) {
            s_f_H[tid * 9 + 3 + k] = H[i][j];
            ++k;
        }
    }

    __syncthreads();
    int t = size_of_block;
    int i,b;
    for(i=t/2, b=(t+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < 9; ++j) {
                s_f_H[tid*9+j] += s_f_H[(tid+b)*9+j];
            }
            __syncthreads();
        }
        i = (b>i) ? b : i;
    }

    if (threadIdx.x == 0) {
        fi = Vector3(0);
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[6];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[7];  H[2][2] = s_f_H[8];
        // symmetry
        H[0][1] = H[1][0];
        H[1][2] = H[2][1];
        H[0][2] = H[2][0];

        // damping (velocity)
        //fi -= damping / dt * H * (ps.p[vid] - ps.last_p[vid]);
        //H  += damping / dt * H;

        // intertia (accellearation)
        const scalar mh2 = ps.m[vid] / (dt*dt);
        fi -= mh2 * (ps.p[vid] - y[vid]);

        r[vid] = fi;
    }
}

std::vector<Vector3> GPU_VBD_FEM::get_forces(const GPU_ParticleSystem *ps, const scalar dt) const {
    for(int c = 0; c < d_thread->nb_kernel; ++c) {
        kernel_vbd_compute_residual<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
            d_thread->nb_threads[c], _damping, dt, d_thread->offsets[c],
             y->buffer, r->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }
    std::vector<Vector3> residual(ps->nb_particles());
    r->get_data(residual);
    return residual;
}


GPU_VBD_FEM::GPU_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping) :
    GPU_FEM(element, geometry, topology, young, poisson, material), y(nullptr)
{
    const int nb_vertices = static_cast<int>(geometry.size());
    _damping = damping;
    d_owners = new GPU_Owners_Data();
    r = new Cuda_Buffer(nb_vertices, Vector3(0.f));
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


void GPU_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937());

    for(const int c : kernels) {
        kernel_vbd_solve<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
            d_thread->nb_threads[c], _damping, dt, d_thread->offsets[c],
             y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }

    const std::vector<Vector3> forces = get_forces(ps, dt);
    scalar residual = 0;
    for(int i = 0; i < forces.size(); ++i) {
        residual += glm::length2(forces[i]);
    }

    DebugUI::Begin("Residual");
    DebugUI::Plot("Plot", residual, 100);
    DebugUI::Value("residual", residual);
    DebugUI::Range("Range", residual);
    DebugUI::End();
}