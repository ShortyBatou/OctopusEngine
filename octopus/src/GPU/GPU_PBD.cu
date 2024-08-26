#include "GPU/GPU_PBD.h"

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
    C = 1e-16f < abs(C) ? 1e-16f : C;;
    scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
    C = sqrt(abs(C)) * s;

    // convert force to constraint gradient
    scalar C_inv = scalar(1.) / scalar(2. * C);
    for (int j = 0; j < nb_vert_elem; ++j) {
        grads[j] *= C_inv;
    }

}

__global__ void kernel_velocity_update(int n, float dt, Vector3* p, Vector3* prev_p, Vector3* v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = (p[i] - prev_p[i]) / dt;
}


__global__ void kernel_step_solver(int n, float dt, Vector3 g, Vector3* p, Vector3* v, Vector3* f, float* w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] += (g + f[i] * w[i])*dt;
    p[i] += v[i] * dt;
    f[i] *= 0;
}



void GPU_PBD_FEM::step(const scalar dt) const {
    kernel_step_solver<<<(cb_position->nb+255)/256, 256>>>(cb_position->nb, dt, Dynamic::gravity(), cb_position->buffer, cb_velocity->buffer, cb_forces->buffer, cb_weights->buffer);
}
