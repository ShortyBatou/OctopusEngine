#include "GPU/VBD/GPU_LF_VBD_FEM.h"

#include <numeric>
#include <random>
#include <GPU/CUMatrix.h>
#include <GPU/GPU_FEM_Material.h>
#include <GPU/Explicit/GPU_Explicit.h>
#include <Manager/TimeManager.h>

__device__ Matrix3x3 snh_lf_elastic_stress(const Matrix3x3 &F, const scalar mu) {
    return mu * (F - mat3x3_com(F));
}

__device__ void snh_lf_elastic_hessian(const Matrix3x3 &F, const scalar mu, Matrix3x3 d2W_dF2[6]) {
    // (d^2_I3 / dx^2)
    d2W_dF2[1] = vec_hat(F[2])  * -mu;
    d2W_dF2[2] = -vec_hat(F[1]) * -mu;
    d2W_dF2[4] = vec_hat(F[0])  * -mu;

    // mu/2 * H2 = mu * I_9x9x
    d2W_dF2[0] = Matrix3x3(mu);
    d2W_dF2[3] = Matrix3x3(mu);
    d2W_dF2[5] = Matrix3x3(mu);
}

__device__ void snh_lf_constraint_hessian(const Matrix3x3 &F, const scalar p, const scalar mu, Matrix3x3 d2W_dF2[6]) {
    // (d^2_I3 / dx^2)
    d2W_dF2[1] = vec_hat(F[2])  * p;
    d2W_dF2[2] = -vec_hat(F[1]) * p;
    d2W_dF2[4] = vec_hat(F[0])  * p;
}

__device__ void snh_lf_constraint_stress(const Matrix3x3 &F, Matrix3x3& P, scalar& C) {
    C = glm::determinant(F) - 1;
    P = mat3x3_com(F);
}


__global__ void kernel_lf_vbd_solve(
    const int n,
    const scalar damping,
    const scalar dt,
    const int offset,
    const Vector3* y,
    scalar* l,
    const scalar* Vi,
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

    // elastic
    Matrix3x3 P = snh_lf_elastic_stress(F, mt.mu);
    Vector3 fi = -P * dF_dx * fem.V[qe_off];
    snh_lf_elastic_hessian(F, mt.mu, d2W_dF2);
    Matrix3x3 K = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2W_dF2);

    // damping (velocity)
    fi -= damping / dt * K * (ps.p[vid] - ps.last_p[vid]);
    K  += damping / dt * K;

    // intertia (accellearation)
    scalar mh2 = ps.m[vid] / dt*dt;
    fi -= mh2 * (ps.p[vid] - y[vid]);
    K[0][0] += mh2; K[1][1] += mh2; K[2][2] += mh2;
    // constraint + constraint gradient
    scalar C = 0;
    snh_lf_constraint_stress(F, P, C);
    Vector3 gradC = Vector3(P * dF_dx) * fem.V[qe_off];
    C *= fem.V[qe_off];

    Matrix3x3 d2C_dF2[6];
    snh_lf_elastic_hessian(F, mt.mu, d2C_dF2);
    Matrix3x3 g2C = assemble_sub_hessian(dF_dx, fem.V[qe_off], d2C_dF2);

    // shared variable : f, H
    //[0-18]
    __shared__ scalar s_f_H[2592]; // size = block_size * 12 * sizeof(float)
    s_f_H[tid * 19 + 6] = C; // [6] constraint
    for(int k = 0, j = 0; j < 3; ++j) {
        s_f_H[tid * 19 + j] = fi[j]; // [0,2] elastic + inertia + damping
        s_f_H[tid * 19 + 3 + j] = gradC[j]; // [3,5] constraint gradient
        for(int i = j; i < 3; ++i) {
            s_f_H[tid * 19 + 7 + k] = K[i][j]; // [7-12] // elastic + intertia + damping + constraint
            s_f_H[tid * 19 + 13 + k] = g2C[i][j]; // [13-18] // elastic + intertia + damping + constraint
            ++k;
        }
    }

    __syncthreads();
    int t = size_of_block;
    for(int i=t/2, k=(t+1)/2; i > 0; k=(k+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < 19; ++j) {
                s_f_H[tid*19+j] += s_f_H[(tid+k)*19+j];
            }
            __syncthreads();
        }
        i = (k>i) ? k : i;
    }

    if (threadIdx.x == 0) {
        fi = Vector3(s_f_H[0], s_f_H[1], s_f_H[2]);
        gradC = Vector3(s_f_H[3], s_f_H[4], s_f_H[5]);
        C = s_f_H[6];
        K[0][0] = s_f_H[7];
        K[1][0] = s_f_H[8]; K[1][1] = s_f_H[10];
        K[2][0] = s_f_H[9]; K[2][1] = s_f_H[11]; K[2][2] = s_f_H[12];
        K[0][1] = K[1][0]; K[1][2] = K[2][1]; K[0][2] = K[2][0];

        g2C[0][0] = s_f_H[13];
        g2C[1][0] = s_f_H[14];  g2C[1][1] = s_f_H[16];
        g2C[2][0] = s_f_H[15];  g2C[2][1] = s_f_H[17]; g2C[2][2] = s_f_H[18];
        g2C[0][1] = g2C[1][0]; g2C[1][2] = g2C[2][1]; g2C[0][2] = g2C[2][0];

        scalar p = l[vid];
        K = glm::inverse(K+g2C*p);
        scalar alpha = Vi[vid] / mt.lambda;
        scalar A = (glm::dot(gradC, K * gradC) + alpha);
        scalar g = alpha * p - C; // constraint
        Vector3 h = fi + p * gradC; // forces
        scalar dt_p = (-g + glm::dot(gradC,  K * h)) / A;
        // we maybe need to update K
        // K = K_I + K_E + K_D + K_C
        // K_C = d²C/dx² p
        Vector3 dt_x = -K * (fi + gradC * (p + dt_p));
        ps.p[vid] += dt_x;
        if(vid == 10) {
            printf("C = %f\n", C);
            printf("dt_p = %f\n", dt_p);
            print_vec(dt_x);
            print_vec(fi - gradC * (p + dt_p));
            print_vec(gradC);
            print_vec(fi);
            print_mat(K);
        }
        l[vid] += dt_p;
    }
}

__global__ void kernel_reset(const int n, scalar* l)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= n) return;
    l[gid] = 0.f;
}

GPU_LF_VBD_FEM::GPU_LF_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping) :
    GPU_VBD_FEM(element, topology, geometry, material, young, poisson, damping)
{
    const int nb_vertices = static_cast<int>(geometry.size());
    l = new Cuda_Buffer<scalar>(nb_vertices, 0.f);

    const FEM_Shape* shape = get_fem_shape(element);
    const int elem_nb_vert = elem_nb_vertices(element);
    const int nb_element = static_cast<int>(topology.size()) / elem_nb_vert;
    const int nb_quadrature = static_cast<int>(shape->weights.size());

    std::vector<scalar> V_vert(geometry.size(),0.f);
    for (int i = 0; i < nb_element; i++) {
        const int id = i * elem_nb_vert;
        scalar V = 0;
        for (int j = 0; j < nb_quadrature; ++j) {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < elem_nb_vert; ++k) {
                J += glm::outerProduct(geometry[topology[id + k]], shape->dN[j][k]);
            }
            V += abs(glm::determinant(J)) * shape->weights[j];
        }
        V /= static_cast<scalar>(elem_nb_vert);
        for (int k = 0; k < elem_nb_vert; ++k) {
            V_vert[topology[id + k]] += V;
        }
    }
    Vi = new Cuda_Buffer<scalar>(V_vert);
}

void GPU_LF_VBD_FEM::start(GPU_ParticleSystem* ps, scalar dt)
{
    const int nb_thread = ps->nb_particles();
    const int block_size = 32;
    const int grid_size = (nb_thread + block_size-1) / block_size;
    kernel_reset<<<grid_size, block_size>>>(nb_thread, l->buffer);
}

void GPU_LF_VBD_FEM::step(GPU_ParticleSystem* ps, const scalar dt) {
    std::vector<int> kernels(d_thread->nb_kernel);
    std::iota(kernels.begin(), kernels.end(), 0);
    std::shuffle(kernels.begin(), kernels.end(), std::mt19937());

    for(const int c : kernels) {
        kernel_lf_vbd_solve<<<d_thread->grid_size[c], d_thread->block_size[c]>>>(
             d_thread->nb_threads[c], damping, dt, d_thread->offsets[c],
             y->buffer, l->buffer, Vi->buffer, *d_material,
             ps->get_parameters(), get_fem_parameters(), get_owners_parameters()
        );
    }
}

