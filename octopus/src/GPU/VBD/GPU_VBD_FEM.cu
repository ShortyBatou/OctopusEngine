#include "GPU/CUMatrix.h"
#include <glm/detail/func_matrix_simd.inl>
#include <Manager/Debug.h>
#include <Manager/Dynamic.h>
#include "GPU/VBD/GPU_VBD_FEM.h"

#include <random>
#include <set>
#include <Tools/Graph.h>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/GPU_ParticleSystem.h>
#include <Manager/Input.h>
#include <Manager/TimeManager.h>

__device__ void compute_f_H(
    const int n, const int r_vid,
    const Material_Data& mt, const GPU_ParticleSystem_Parameters ps, const int* topo,
    const Matrix3x3 &JX_inv, const scalar V, const Vector3* dN,
    Vector3& fi, Matrix3x3& H) {
    Matrix3x3 Jx(0.f);
    Matrix3x3 d2W_dF2[9];

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


__device__ void compute_f_H_sym(
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
    eval_hessian_sym(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    H += assemble_sub_hessian_sym(dF_dx, V, d2W_dF2);
}


__device__ void vec_reduction(const int tid, const int block_size, const int offset, const int v_size, scalar* s_data) {
    __syncthreads();
    int i,b;
    for(i=block_size/2, b=(block_size+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < v_size; ++j) {
                s_data[(offset + tid)*v_size+j] += s_data[(offset + tid+b)*v_size+j];
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
    vec_reduction(tid, block_size, 0, 12, s_f_H); // reduction of fi Hi

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
    vec_reduction(tid, block_size, 0, 12, s_f_H); // reduction of fi Hi

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
    compute_f_H_sym(fem.elem_nb_vert, r_vid,
                mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                fi, H);


    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 9 * sizeof(float)
    store_f_H_in_shared_sym(tid, fi, H, s_f_H);
    vec_reduction(tid, size_of_block, 0, 9, s_f_H);

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

__global__ void kernel_vbd_solve_v4(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners,
    GPU_BLock_Parameters blocks
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // sub block system
    const int bid = offset + blockIdx.x; // block id
    const int b_sub_size = blocks.sub_block_size[bid];
    const int b_nb_sub_block = blocks.nb_sub_block[bid];
    const int b_size = b_sub_size * b_nb_sub_block;
    const int b_sub_id = threadIdx.x / b_sub_size;
    if(threadIdx.x >= b_size) return;


    const int tid = threadIdx.x % b_sub_size;
    // vertex information
    const int cid = blocks.offset[bid] + b_sub_id;
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
    compute_f_H_sym(fem.elem_nb_vert, r_vid,
                mt, ps, topo, fem.JX_inv[qe_off], fem.V[qe_off], fem.dN + qv_off,
                fi, H);


    // shared variable : f, H
    extern __shared__ scalar s_f_H[]; // size = block_size * 9 * sizeof(float)
    const int s_off = b_sub_id * b_sub_size;
    const int o = s_off * 9;
    store_f_H_in_shared_sym(tid, fi, H, s_f_H+o);
    vec_reduction(tid, b_sub_size, s_off, 9, s_f_H);

    if (tid == 0) {
        fi.x = s_f_H[o + 0]; fi.y = s_f_H[o + 1]; fi.z = s_f_H[o + 2];
        H[0][0] = s_f_H[o + 3];
        H[1][0] = s_f_H[o + 4]; H[1][1] = s_f_H[o + 6];
        H[2][0] = s_f_H[o + 5]; H[2][1] = s_f_H[o + 7];  H[2][2] = s_f_H[o + 8];
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
    vec_reduction(tid, size_of_block, 0, 9, s_f_H);

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
        scalar s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
        kernel_vbd_compute_residual<<<d_thread->grid_size[c] * d_fem->nb_quadrature, d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
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
    d_blocks = new GPU_Block_Data();
    r = new Cuda_Buffer(nb_vertices, Vector3(0.f));
    std::vector<std::vector<int>> e_owners;
    std::vector<std::vector<int>> e_ref_id;
    build_graph_color(element, topology, nb_vertices, _colors,e_owners,e_ref_id);
    sort_by_color(nb_vertices, _colors, e_owners, e_ref_id);
}

void GPU_VBD_FEM::sort_by_color(const int nb_vertices, const std::vector<int>& colors, const std::vector<std::vector<int>>& e_owners, const std::vector<std::vector<int>>& e_ref_id)
{
    // sort by color
    std::vector<int> ref_id;
    std::vector<int> owners;
    std::vector<int> nb_owners;
    std::vector<int> owners_offset;

    std::vector<int> nb_sub_block;      // nb sub_block in block
    std::vector<int> sub_block_size;    // size of one sub block
    std::vector<int> block_offset;      // first vertice id in block

    int total_thread = 0;
    int total_merge_thread = 0;
    // sort neighbors
    for(int c = 0; c < d_thread->nb_kernel; ++c) {
        int n_max = 1; // max block size
        int nb_block = 0; // nb vert in color
        int offset = static_cast<int>(version == Block_Merge ? nb_sub_block.size() : nb_owners.size()); // nb vertices
        std::vector<int> ids; // block id
        for(int i = 0; i < nb_vertices; ++i) {
            if(c != colors[i]) continue;
            ids.push_back(i);
            n_max = std::max(n_max, static_cast<int>(e_owners[i].size()));
            nb_block++;
        }

        total_thread += nb_block * n_max;
        // the max block size depends on the largest block in color and needs to be a multiple of 32 (NVidia)
        int vmax = n_max;
        if(version == Block_Merge) {
            // on devrait avoir une taille par bloc !
            // s'il y a  32 thread qui font rien, le bloc est terminé extrèmement vite
            vmax = (n_max * d_fem->nb_quadrature / 32 + (n_max * d_fem->nb_quadrature % 32 == 0 ? 0 : 1)) * 32;
            nb_block = 0;
            // sort id by nb_owners
            std::sort(ids.begin(), ids.end(),
                 [&](const int& a, const int& b) {
                     return e_owners[a].size() < e_owners[b].size();
                 }
            );

            // create the block merge data
            std::cout << "C = " << c << "   MAX = " << vmax << std::endl;
            int off = nb_owners.size();
            for(int i = 0; i < ids.size(); ++i) {
                const int v = e_owners[ids[i]].size() * d_fem->nb_quadrature; // block size
                // nb block that have the same size
                int nb = 1;
                while(i+1 < ids.size() && e_owners[ids[i+1]].size() * d_fem->nb_quadrature == v) {
                    nb++; i++;
                }
                // how we merge the block depending on the max block size
                const int max_block = vmax / v; // max size sub_group
                const int nb_group = nb / max_block; // how many sub group
                const int rest = (nb - max_block * nb_group) * v;  // nb thread that can't be totally merge
                for(int j = 0; j < nb_group; ++j) {
                    nb_sub_block.push_back(max_block);
                    sub_block_size.push_back(v);
                    block_offset.push_back(off);
                    off += max_block;
                    nb_block++;
                }
                if(rest != 0) {
                    nb_sub_block.push_back(nb - max_block * nb_group);
                    sub_block_size.push_back(v);
                    block_offset.push_back(off);
                    off += nb - max_block * nb_group;
                    nb_block++;
                }
                std::cout << v << " x " << nb << " => " << nb_group << "x(" << max_block << "x" << v << ") + (" << rest / v << "x" << v << ")" << std::endl;
            }
            total_merge_thread += n_max * nb_block;
            std::cout << std::endl;
            vmax /= d_fem->nb_quadrature;
        }

        for(const int id : ids) {
            nb_owners.push_back(static_cast<int>(e_owners[id].size()));
            owners_offset.push_back(static_cast<int>(owners.size()));
            owners.insert(owners.end(), e_owners[id].begin(), e_owners[id].end());
            ref_id.insert(ref_id.end(), e_ref_id[id].begin(), e_ref_id[id].end());
        }

        d_thread->grid_size.push_back(nb_block);
        d_thread->nb_threads.push_back(nb_block * vmax);
        d_thread->offsets.push_back(offset);
        d_thread->block_size.push_back(vmax);
    }
    std::cout << total_thread << " vs " << total_merge_thread << std::endl;
    //vert data
    d_owners->cb_nb = new Cuda_Buffer(nb_owners);
    d_owners->cb_offset = new Cuda_Buffer(owners_offset);

    // element data
    d_owners->cb_eids = new Cuda_Buffer(owners);
    d_owners->cb_ref_vid = new Cuda_Buffer(ref_id);

    // block data
    d_blocks->cb_nb_sub_block = new Cuda_Buffer<int>(nb_sub_block);
    d_blocks->cb_offset = new Cuda_Buffer<int>(block_offset);
    d_blocks->cb_sub_block_size = new Cuda_Buffer<int>(sub_block_size);
}


void GPU_VBD_FEM::build_graph_color(const Element element, const Mesh::Topology &topology, const int nb_vertices,
    std::vector<int> &colors, std::vector<std::vector<int>>& e_neighbors, std::vector<std::vector<int>>& e_ref_id)
{
    e_neighbors.resize(nb_vertices);
    e_ref_id.resize(nb_vertices);
    // for each vertice get all its neighboors
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        int eid = i / d_fem->elem_nb_vert;
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            e_neighbors[topology[i + j]].push_back(eid);
            e_ref_id[topology[i+j]].push_back(j);
        }
    }
    t_neighbors = e_neighbors;

    Time::Tic();
    p_graph = new Graph(element, topology);
    p_graph->save_as_file("graph");
    std::cout << "P Graph :" << Time::Tac() << std::endl;
    d_graph = new Graph(element, topology, false);
    std::cout << "D Graph :" << Time::Tac() << std::endl;
    Time::Tic();
    Coloration coloration = version >= Better_Coloration ? GraphColoration::DSAT(*p_graph) : GraphColoration::DSAT(*p_graph);
    std::cout << "Coloration " << Time::Tac() << std::endl;
    Time::Tic();
    //Coloration c2 = GraphColoration::Primal_Dual_Element(element, topology, *p_graph, *d_graph);
    Coloration c2 = coloration;
    std::cout << "Test " << Time::Tac() << std::endl;
    _t_nb_color = c2.nb_color;
    _t_color = c2.color;
    _t_conflict = GraphColoration::Get_Conflict(*p_graph, c2);

    /*Time::Tic();
    Coloration c3 = GraphColoration::BFS(*p_graph);
    std::cout << "BFS " << Time::Tac() << std::endl;

    Time::Tic();
    Coloration c4 = GraphColoration::Greedy(*p_graph);
    std::cout << "Greedy " << Time::Tac() << std::endl;

    Time::Tic();
    Coloration c6 = GraphColoration::Greedy_RLF(*p_graph);
    std::cout << "Greedy RLF" << Time::Tac() << std::endl;

    Time::Tic();
    Coloration c7 = GraphColoration::Greedy_SLF(*p_graph);
    std::cout << "Greedy SLF " << Time::Tac() << std::endl;

    Time::Tic();
    Coloration c8 = GraphColoration::DSAT(*p_graph);
    std::cout << "DSAT " << Time::Tac() << std::endl;

    Time::Tic();
    Coloration c9 = GraphColoration::Primal_Dual_DSAT(element, topology, *p_graph, *d_graph);
    std::cout << "PD DSAT " << Time::Tac() << std::endl;

    std::cout << "Test " << c2.nb_color << std::endl;
    std::cout << "Coloration " << coloration.nb_color << std::endl;
    std::cout << "BFS " << c3.nb_color << std::endl;
    std::cout << "Greedy " << c4.nb_color << std::endl;
    std::cout << "Greedy RLF " << c6.nb_color << std::endl;
    std::cout << "Greedy SLF " << c7.nb_color << std::endl;
    std::cout << "DSAT " << c8.nb_color << std::endl;
    std::cout << "PD DSAT " << c9.nb_color << std::endl;*/

    //coloration.nb_color += 2;
    //GraphBalance::Greedy(*p_graph, coloration, 100000);

    colors = coloration.color;

    d_thread->nb_kernel = coloration.nb_color;

    std::cout << "NB color: " << d_thread->nb_kernel << std::endl;
    std::vector<int> nb_per_color(coloration.nb_color, 0);
    for(const int c : colors) {
        nb_per_color[c]++;
    }
    for(int i = 0; i < coloration.nb_color; ++i) {
        std::cout << "c " << i << " = " <<  nb_per_color[i] << " verts" << std::endl;
    }
}

void GPU_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937(seed));
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
            case Reduction_Symmetry : case Better_Coloration :
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
                kernel_vbd_solve_v3<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                    y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
                );
            break;
            case Block_Merge :
                s = d_thread->block_size[c] * d_fem->nb_quadrature * 9 * sizeof(scalar);
                kernel_vbd_solve_v4<<<d_thread->grid_size[c], d_thread->block_size[c] * d_fem->nb_quadrature, s>>>(
                    d_thread->nb_threads[c] * d_fem->nb_quadrature, damping, dt, d_thread->offsets[c],
                    y->buffer, *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters(), get_block_parameters()
                );
            break;
        }
    }
    /*
    std::vector<Vector3> positions(d_graph->n);
    ps->get_position(positions);
    std::vector<int> topo(d_fem->cb_topology->nb);
    d_fem->cb_topology->get_data(topo);


    Debug::SetColor(ColorBase::Green());
    static int v = 1;
    if(Input::Down(Key::M)) v++;
    
    // display all non colored vertices
    if(Input::Loop(Key::W)) {
        Debug::SetColor(ColorBase::Black());
        for(int i = 0; i < positions.size(); i++) {
            if(_t_color[i] == -1) {
                Debug::Cube(positions[i], 0.001);
            }
        }
    }
    if(Input::Loop(Key::X)) {
        ColorMap::Set_Type(ColorMap::Rainbow);
        for(int i = 0; i < positions.size(); i++) {
            if(_t_color[i] != -1) {
                const scalar t = static_cast<scalar>(_t_color[i]) / static_cast<scalar>(_t_nb_color);
                Color c = ColorMap::evaluate(t);
                Debug::SetColor(c);
            }
            else {
                Debug::SetColor(ColorBase::Black());
            }
            Debug::Cube(positions[i], 0.01);
        }
    }

    // display bad elements
    if(Input::Loop(Key::C) || Input::Loop(Key::N)) {
        std::vector<int> edges = ref_edges(get_elem_by_size(d_fem->elem_nb_vert));
        for(int eid = 0; eid < d_graph->n; ++eid) {
            std::set<int> neighbors;
            for(int j : d_graph->adj[eid]) {
                neighbors.insert(j);
            }
            bool bad = false;
            for(int j : d_graph->adj[eid]) {
                for(int k : d_graph->adj[j]) {
                    if(neighbors.find(k) != neighbors.end()) {
                        bad = true;
                        break;
                    }
                }
                if(bad) break;
            }

            if(bad) {
                if(Input::Loop(Key::C)) {
                    Debug::SetColor(ColorBase::Red());
                    for(int i = 0; i < edges.size(); i+=2) {
                        int a = topo[eid * d_fem->elem_nb_vert + edges[i]];
                        int b = topo[eid * d_fem->elem_nb_vert + edges[i+1]];
                        Debug::Line(positions[a],positions[b]);
                    }
                }
                else {
                    Vector3 p = Vector3(0, 0, 0);
                    for(int i = 0; i < d_fem->elem_nb_vert; ++i) {
                        p += positions[topo[eid * d_fem->elem_nb_vert + i]];
                    }
                    p/= static_cast<scalar>(d_fem->elem_nb_vert);
                    for(int j : d_graph->adj[eid]) {
                        Vector3 p2 = Vector3(0, 0, 0);
                        for(int k = 0; k < d_fem->elem_nb_vert; ++k) {
                            p2 += positions[topo[j * d_fem->elem_nb_vert + k]];
                        }
                        p2 /= static_cast<scalar>(d_fem->elem_nb_vert);
                        Debug::Line(p, p2);
                    }
                }
            }
        }
    }

    // display conflict in coloration
    if(Input::Loop(Key::V)) {
        for(auto [vid, nb] : _t_conflict) {
            if(_t_color[vid] != -1) {
                const scalar t = static_cast<scalar>(_t_color[vid]) / static_cast<scalar>(_t_nb_color);
                Color c = ColorMap::evaluate(t);
                Debug::SetColor(c);
            }
            else {
                Debug::SetColor(ColorBase::Black());
            }

            Debug::Cube(positions[vid], 0.005);

            for(int j : p_graph->adj[vid]) {
                if(_t_conflict.find(j) == _t_conflict.end()) continue;
                if(_t_color[j] != _t_color[vid]) continue;
                Debug::Line(positions[vid], positions[j]);
            }
        }
    }

    // display graph dual
    if(Input::Loop(Key::B)) {
        Debug::SetColor(ColorBase::Red());

        for(int eid = 0; eid < d_graph->n; ++eid) {
            Vector3 p = Vector3(0);
            bool colored = false;
            for(int i = 0; i < d_fem->elem_nb_vert; ++i) {
                p += positions[topo[eid * d_fem->elem_nb_vert + i] ];
                if(_t_color[topo[eid * d_fem->elem_nb_vert + i]] != -1) colored = true;
            }
            if(!colored) continue;

            p /= static_cast<scalar>(d_fem->elem_nb_vert);
            for(int eid2 : d_graph->adj[eid]) {
                Vector3 p2 = Vector3(0);
                for(int i = 0; i < d_fem->elem_nb_vert; ++i)
                    p2 += positions[topo[eid2 * d_fem->elem_nb_vert+ i] ];
                p2 /= static_cast<scalar>(d_fem->elem_nb_vert);
                Debug::Line(p, p2);
            }
        }
    }/**/
}