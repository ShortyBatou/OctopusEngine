#include "GPU/CUMatrix.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include "GPU/VBD/GPU_VBD_FEM.h"

#include <random>
#include <set>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/GPU_ParticleSystem.h>




__device__ void compute_f_H(
    const int n, const int r_vid,
    const Material_Data& mt, const GPU_ParticleSystem_Parameters ps, const int* topo,
    const Matrix3x3 &JX_inv, const scalar V, const Vector3* dN,
    Vector3& fi, Matrix3x3& H) {
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[6];

    for (int i = 0; i < n; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], dN[i]);
    }

    const Matrix3x3 F = Jx * JX_inv;
    const Vector3 dF_dx = glm::transpose(JX_inv) * dN[r_vid];

    // Compute force at vertex i
    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
    fi -= P * dF_dx * V;

    // Compute hessian
    eval_hessian(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    H += assemble_sub_hessian(dF_dx, V, d2W_dF2);
}

__device__ void vec_reduction(const int tid, const int block_size, const int v_size, scalar* s_data) {
    __syncthreads();
    int i,b;
    for(i=block_size/2, b=(block_size+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < v_size; ++j) {
                s_data[tid*v_size+j] += s_data[(tid+b)*v_size+j];
            }
            __syncthreads();
        }
        i = (b>i) ? b : i;
    }
}

__device__ void store_f_H_in_shared(const int tid, const Vector3& fi, const Matrix3x3& H, scalar* s_data) {
    int k = 0;
    for(int j = 0; j < 3; ++j) {
        s_data[tid * 12 + j] = fi[j];
        for(int i = 0; i < 3; ++i) {
            s_data[tid * 12 + 3 + k] = H[i][j];
            ++k;
        }
    }
}

__device__ void store_f_H_in_shared_sym(const int tid, const Vector3& fi, const Matrix3x3& H, scalar* s_data) {
    int k = 0;
    for(int j = 0; j < 3; ++j) {
        s_data[tid * 9 + j] = fi[j];
        for(int i = j; i < 3; ++i) {
            s_data[tid * 9 + 3 + k] = H[i][j];
            ++k;
        }
    }
}


__device__ Vector3 compute_correction(const int vid, const scalar damping, const scalar dt,
        GPU_ParticleSystem_Parameters ps, const Vector3* y,
        Vector3& fi, Matrix3x3& H) {
    // damping (velocity)
    fi -= damping / dt * H * (ps.p[vid] - ps.last_p[vid]);
    H  += damping / dt * H;

    // inertia (acceleration)
    const scalar mh2 = ps.m[vid] / (dt*dt);
    fi -= mh2 * (ps.p[vid] - y[vid]);
    H[0][0] += mh2; H[1][1] += mh2; H[2][2] += mh2;
    //scalar detH = glm::determinant(s_H);
    const scalar detH = abs(glm::determinant(H));
    return detH > 1e-6f ? glm::inverse(H) * fi : Vector3(0.f);
}

__global__ void kernel_vbd_solve_v1(
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
    if (blockIdx.x * blockDim.x + threadIdx.x >= n) return;

    // the group size depends on the number of element that contains these vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
    const int block_size = owners.nb[cid];
    const int tid = threadIdx.x;
    if (tid >= block_size) return;

    const int e_off = owners.offset[cid] + tid; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];

    if(ps.mask[vid] == 0) return;

    Vector3 fi(0.f);
    Matrix3x3 H(0.f);
    for(int q = 0; q < fem.nb_quadrature; ++q) {
        const int qe_off = eid * fem.nb_quadrature + q;
        const int qv_off = q * fem.elem_nb_vert;
        compute_f_H(fem.elem_nb_vert, r_vid,
                    mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                    fi, H);
    }

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 12 * sizeof(float)
    store_f_H_in_shared(tid, fi, H, s_f_H); // store f and h in shared memory
    vec_reduction(tid, block_size, 12, s_f_H); // reduction of fi Hi

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3]; H[0][1] = s_f_H[6]; H[0][2] = s_f_H[9 ];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[7]; H[1][2] = s_f_H[10];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[8]; H[2][2] = s_f_H[11];

        ps.p[vid] += compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}

__global__ void kernel_vbd_solve_v2(
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

    // the group size depends on the number of element that contains these vertices
    // and the number of needed quadratures
    const int cid = offset + blockIdx.x; // vertex position in coloration
    const int block_size = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= block_size) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    if(ps.mask[vid] == 0) return;

    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;

    Vector3 fi(0.f);
    Matrix3x3 H(0.f);

    compute_f_H(fem.elem_nb_vert, r_vid,
                    mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                    fi, H);

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 12 * sizeof(float)
    store_f_H_in_shared(tid, fi, H, s_f_H); // store f and h in shared memory
    vec_reduction(tid, block_size, 12, s_f_H); // reduction of fi Hi

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3]; H[0][1] = s_f_H[6]; H[0][2] = s_f_H[9 ];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[7]; H[1][2] = s_f_H[10];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[8]; H[2][2] = s_f_H[11];

        ps.p[vid] += compute_correction(vid, damping, dt, ps, y, fi, H);
    }
}

__global__ void kernel_vbd_solve_v3(
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

    // the group size depends on the number of element that contains these vertices
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

    Vector3 fi(0.f);
    Matrix3x3 H(0.f);
    compute_f_H(fem.elem_nb_vert, r_vid,
                mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                fi, H);


    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 9 * sizeof(float)
    store_f_H_in_shared_sym(tid, fi, H, s_f_H);
    vec_reduction(tid, size_of_block, 9, s_f_H);

    if (threadIdx.x == 0) {
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[6];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[7];  H[2][2] = s_f_H[8];
        // symmetry
        H[0][1] = H[1][0]; H[1][2] = H[2][1]; H[0][2] = H[2][0];

        ps.p[vid] += compute_correction(vid, damping, dt, ps, y, fi, H);
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

    // the group size depends on the number of element that contains these vertices
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
    Vector3 fi(0.f);
    Matrix3x3 H(0.f);
    compute_f_H(fem.elem_nb_vert, r_vid,
                mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                fi, H);

    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 12 * sizeof(float)
    store_f_H_in_shared_sym(tid, fi, H, s_f_H);
    vec_reduction(tid, size_of_block, 9, s_f_H);

    if (threadIdx.x == 0) {
        fi = Vector3(0);
        fi.x = s_f_H[0]; fi.y = s_f_H[1]; fi.z = s_f_H[2];
        H[0][0] = s_f_H[3];
        H[1][0] = s_f_H[4]; H[1][1] = s_f_H[6];
        H[2][0] = s_f_H[5]; H[2][1] = s_f_H[7];  H[2][2] = s_f_H[8];
        // symmetry
        H[0][1] = H[1][0]; H[1][2] = H[2][1]; H[0][2] = H[2][0];

        // damping (velocity)
        fi -= damping / dt * H * (ps.p[vid] - ps.last_p[vid]);
        H  += damping / dt * H;

        // intertia (accellearation)
        const scalar mh2 = ps.m[vid] / (dt*dt);
        fi -= mh2 * (ps.p[vid] - y[vid]);
        H  += mh2 * Matrix3x3(1.);
        r[vid] = fi;
    }
}

std::vector<Vector3> GPU_VBD_FEM::get_forces(const GPU_ParticleSystem *ps, const scalar dt) const {
    for(int c = 0; c < d_thread->nb_kernel; ++c) {
        kernel_vbd_compute_residual<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
            d_thread->nb_threads[c], damping, dt, d_thread->offsets[c],
             y->buffer, r->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }
    std::vector<Vector3> residual(ps->nb_particles());
    r->get_data(residual);
    return residual;
}


GPU_VBD_FEM::GPU_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping,
                         const VBD_Version& v) :
    GPU_FEM(element, geometry, topology, young, poisson, material), version(v), y(nullptr)
{
    const int nb_vertices = static_cast<int>(geometry.size());
    this->damping = damping;
    d_owners = new GPU_Owners_Data();
    r = new Cuda_Buffer(nb_vertices, Vector3(0.f));
    std::vector<std::vector<int>> e_owners;
    std::vector<std::vector<int>> e_ref_id;
    build_graph_color(topology, nb_vertices, _colors,e_owners,e_ref_id);
    sort_by_color(nb_vertices, _colors, e_owners, e_ref_id);
    shared_sizes.push_back(10);
}

void GPU_VBD_FEM::sort_by_color(const int nb_vertices, const std::vector<int>& colors, const std::vector<std::vector<int>>& e_owners, const std::vector<std::vector<int>>& e_ref_id)
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
        d_thread->block_size.push_back(n_max);
        shared_sizes.push_back(n_max * sizeof(scalar));
        d_thread->nb_threads.push_back(nb_vert * n_max);
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
    unsigned int s;
    for(const int c : kernels) {
        switch(version) {
            case Base :
                s = d_thread->block_size[c] * 12 * sizeof(scalar);
                kernel_vbd_solve_v1<<<d_thread->grid_size[c], d_thread->block_size[c], s>>>(
                    d_thread->nb_threads[c], damping, dt, d_thread->offsets[c],
                      y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters());
            break;
            case Threaded_Quadrature :
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 12 * sizeof(scalar);
                kernel_vbd_solve_v2<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                     y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
                );
            break;
            case Reduction_Symmetry :
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
                kernel_vbd_solve_v3<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                    y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
                );
            break;
        }
    }
}