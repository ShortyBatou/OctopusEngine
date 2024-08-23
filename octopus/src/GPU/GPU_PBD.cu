#include "GPU/GPU_PBD.h"
#include <Dynamic/FEM/FEM_Shape.h>
#include <Manager/Dynamic.h>
#include "Core/Base.h"

__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3* pos, Vector3* dN) {
    // Compute transform (reference => scene)
    Matrix3x3 Jx = Matrix3x3(0.); ;
    for(int j = 0; j < nb_vert_elem; ++j) {
        Jx = glm::outerProduct(pos[j], dN[nb_vert_elem + j]);
    }

    return Jx;
}

__global__ void kernel_constraint_solve(int n, int nb_quadrature, int nb_vert_elem, int offset, Vector3* p, int* topology, Vector3* dN, scalar* V, Matrix3x3* JX_inv) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x; // num of element
    if (eid >= n) return;
    int vid = offset + eid * nb_vert_elem; // first vertice id in topology
    int qid = eid * nb_quadrature;

    scalar C = 0., energy = 1.;
    Vector3 pos[32];
    Vector3 grads[32];

    // get position
    for(int i = 0; i < nb_vert_elem; ++i) {
        pos[i] = p[vid + i];
    }

    // evaluate constraint and gradients
    Matrix3x3 P;
    for (int i = 0; i < nb_quadrature; ++i) {
        // Deformation gradient (material => scene   =   material => reference => scene)
        Matrix3x3 F = compute_transform(nb_vert_elem, pos, dN+i*nb_vert_elem) * JX_inv[i*nb_vert_elem];
        // Get piola kirchoff stress tensor + energy

        // add forces
        P = P * glm::transpose(JX_inv[qid + i]) * V[qid + i];
        for (int j = 0; j < nb_vert_elem; ++j)
            grads[j] += P * dN[i*nb_vert_elem + j];

        // add energy
        C += energy * V[eid * nb_quadrature + i];
    }

    // convert energy to constraint
    C = min(1e-16, abs(C));
    scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
    C = sqrt(abs(C)) * s;

    // convert force to constraint gradient
    scalar C_inv = scalar(1.) / scalar(2. * C);
    for (int j = 0; j < nb_vert_elem; ++j) {
        grads[j] *= C_inv;
    }

}

__global__ void kernel_velocity_update(int n, float dt, Vector3* p, Vector3* prev_p, Vector3* v) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = (p[i] - prev_p[i]) / dt;
}


__global__ void kernel_step_solver(int n, float dt, Vector3 g, Vector3* p, Vector3* v, Vector3* f, float* w) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] += (g + f[i] * w[i])*dt;
    p[i] += v[i] * dt;
    f[i] *= 0;
}


GPU_PB_FEM::GPU_PB_FEM(Element element, const std::vector<Vector3>& geometry, const std::vector<int>& topology, const std::vector<int>& offsets, float density) {
    FEM_Shape* shape = get_fem_shape(element); shape->build();
    int elem_nb_vert = elem_nb_vertices(element);
    nb_verts = static_cast<int>(geometry.size());
    nb_elem = static_cast<int>(topology.size()) / elem_nb_vert;
    nb_quadrature = static_cast<int>(shape->weights.size());

    std::vector<scalar> mass(geometry.size());
    std::vector<scalar> inv_mass(mass.size());
    std::vector<Vector3> dN(nb_quadrature * elem_nb_vert);
    std::vector<Matrix3x3> JX_inv(nb_quadrature * nb_elem);
    std::vector<scalar> V(nb_quadrature * nb_elem);


    for (int i = 0; i < nb_quadrature; i++) {
        dN.insert(dN.begin() + i * elem_nb_vert, shape->dN[i].begin(),  shape->dN[i].end() );
    }


    for (int i = 0; i < nb_elem; i++) {
        scalar V_sum = 0;
        const int id = i * elem_nb_vert;
        const int eid = i * nb_quadrature;
        for (int j = 0; j < nb_quadrature; ++j) {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < elem_nb_vert; ++k) {
                J += glm::outerProduct(geometry[topology[id+k]], dN[j * elem_nb_vert + k]);
            }
            V[eid+j] = abs(glm::determinant(J)) * shape->weights[j];
            JX_inv[eid+j] = glm::inverse(J);
            V_sum += V[eid+j];
        }

        for (int j = 0; j < elem_nb_vert; ++j) {
            mass[topology[id + j]] += density * V_sum / static_cast<scalar>(elem_nb_vert);
        }
    }

    for (int i = 0; i < mass.size(); ++i) {
        inv_mass[i] = 1.f / mass[i];
    }

    cb_position = new Cuda_Buffer(geometry);
    cb_prev_position = new Cuda_Buffer(geometry);
    cb_velocity = new Cuda_Buffer(std::vector(nb_verts, Unit3D::Zero()));
    cb_forces = new Cuda_Buffer(std::vector(nb_verts, Unit3D::Zero()));
    cb_mass = new Cuda_Buffer(mass);
    cb_inv_mass = new Cuda_Buffer(inv_mass);

    cb_topology = new Cuda_Buffer(topology);
    cb_offsets = new Cuda_Buffer(offsets);
    cb_weights = new Cuda_Buffer<scalar>(shape->get_weights());
    cb_dN = new Cuda_Buffer(dN);
    cb_V = new Cuda_Buffer(V);
    cb_JX_inv = new Cuda_Buffer(JX_inv);
}

void GPU_PB_FEM::step(scalar dt) {
    kernel_step_solver<<<(cb_position->nb+255)/256, 256>>>(cb_position->nb, dt, Dynamic::gravity(), cb_position->buffer, cb_velocity->buffer, cb_forces->buffer, cb_weights->buffer);

}

GPU_PB_FEM::~GPU_PB_FEM() {
    delete cb_position;
    delete cb_prev_position;
    delete cb_velocity;
    delete cb_forces;
    delete cb_mass;
    delete cb_inv_mass;

    delete cb_topology;
    delete cb_offsets;
    delete cb_JX_inv;
    delete cb_dN;
    delete cb_weights;
    delete cb_V;
}