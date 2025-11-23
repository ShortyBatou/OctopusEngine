#include "GPU/Explicit/GPU_FEM_Explicit.h"
#include "GPU/GPU_FEM_Material.h"
#include <GPU/CUMatrix.h>

__global__ void kernel_fem_eval_force(
    // nb_thread, nb quadrature per elements, nb vertices in element
    const int n, const scalar damping,
    const Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    GPU_Owners_Parameters owners
) {
    // global id
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // the group size depends on the number of element that contains this vertices
    // and the number of needed quadratures
    const int cid = blockIdx.x; // vertex position
    const int size_of_block = owners.nb[cid] * fem.nb_quadrature;
    const int tid = threadIdx.x; // thread id in block
    if (tid >= size_of_block) return;

    const int qid = tid % fem.nb_quadrature; // quadrature number
    const int e_off = owners.offset[cid] + tid / fem.nb_quadrature; // offset in buffer to find the right element
    const int eid = owners.eids[e_off]; // element id
    const int r_vid = owners.ref_vid[e_off]; // vertex id in this element

    const int *topo = fem.topology + eid * fem.elem_nb_vert; // offset the pointer at the start of the element's topology
    const int vid = topo[r_vid];
    const int qe_off = eid * fem.nb_quadrature + qid;
    const int qv_off = qid * fem.elem_nb_vert;

    Matrix3x3 Jx(0.f);

    for (int i = 0; i < fem.elem_nb_vert; ++i) {
        Jx += glm::outerProduct(ps.p[topo[i]], fem.dN[qv_off + i]);
    }
    const Matrix3x3 F = Jx * fem.JX_inv[qe_off];

    const Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);

    const Vector3 dF_dx = glm::transpose(fem.JX_inv[qe_off]) * fem.dN[qv_off + r_vid];
    // Compute force at vertex i
    Vector3 fi = -P * dF_dx * fem.V[qe_off];

    // assemble hessian
    Matrix3x3 d2W_dF2[6];
    eval_hessian(mt.material, mt.lambda, mt.mu, F, d2W_dF2);
    const Matrix3x3 H = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2W_dF2);

    //Matrix3x3 H2 = glm::outerProduct(fi, fi);
    fi -= damping * H * ps.v[vid];/**/

    // shared variable : f, H
    __shared__ Vector3 s_f_H[256]; // size = block_size * 3 * sizeof(float)
    s_f_H[tid] = fi;

    __syncthreads();
    const int t = size_of_block;
    int i,b;
    for(i=t/2, b=(t+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            s_f_H[tid] += s_f_H[tid+b];
            __syncthreads();
        }
        i = (b>i) ? b : i;
    }

    if (threadIdx.x == 0) {
        ps.f[vid] = Vector3(s_f_H[0]);
    }
}

GPU_FEM_Explicit::GPU_FEM_Explicit(const Element element, const Mesh::Geometry& geometry, const Mesh::Topology& topology, // mesh
                                   const scalar young, const scalar poisson, const Material material, const scalar damping)
    : GPU_FEM(element, geometry, topology, young, poisson, material), _damping(damping)
{
    const int nb_vertices = static_cast<int>(geometry.size());
    std::vector<std::vector<int>> e_owners(nb_vertices);
    std::vector<std::vector<int>> e_ref_id(nb_vertices);
    // for each vertice get all its neighboors
    for (int i = 0; i < topology.size(); i += d_fem->elem_nb_vert) {
        for (int j = 0; j < d_fem->elem_nb_vert; ++j) {
            e_owners[topology[i + j]].push_back(i / d_fem->elem_nb_vert);
            e_ref_id[topology[i+j]].push_back(j);
        }
    }
    // sort by color
    std::vector<int> ref_id;
    std::vector<int> owners;
    std::vector<int> nb_owners;
    std::vector<int> offset;

    // sort neighbors
    int n_max = 1;

    for(int i = 0; i < nb_vertices; ++i) {
        offset.push_back(static_cast<int>(owners.size()));
        owners.insert(owners.end(), e_owners[i].begin(), e_owners[i].end());
        ref_id.insert(ref_id.end(), e_ref_id[i].begin(), e_ref_id[i].end());
        nb_owners.push_back(static_cast<int>(e_owners[i].size()));
        n_max = std::max(n_max, nb_owners.back());
    }

    d_thread->block_size.push_back(n_max * d_fem->nb_quadrature);
    d_thread->nb_threads.push_back(nb_vertices * d_thread->block_size[0]);
    d_thread->grid_size.push_back(nb_vertices);
    d_thread->offsets.push_back(0);
    std::cout << "EXPLICIT NB THREAD = " << nb_vertices * d_thread->block_size[0] << std::endl;
    d_owners = new GPU_Owners_Data();
    d_owners->cb_nb = new Cuda_Buffer(nb_owners);
    d_owners->cb_eids = new Cuda_Buffer(owners);
    d_owners->cb_offset = new Cuda_Buffer(offset);
    d_owners->cb_ref_vid = new Cuda_Buffer(ref_id);
}

void GPU_FEM_Explicit::step(GPU_ParticleSystem* ps, scalar dt)
{
    kernel_fem_eval_force<<<d_thread->grid_size[0], d_thread->block_size[0]>>>(
        d_thread->nb_threads[0], _damping,
        *d_material, ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
    );
}

