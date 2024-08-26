#include "GPU/GPU_PBD.h"

__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3* pos, Vector3* dN) {
    // Compute transform (reference => scene)
    Matrix3x3 Jx = Matrix3x3(0.); ;
    for(int j = 0; j < nb_vert_elem; ++j) {
        Jx = glm::outerProduct(pos[j], dN[nb_vert_elem + j]);
    }

    return Jx;
}

__global__ void kernel_constraint_solve(
    const int n, const int nb_quadrature, const int nb_vert_elem, const int offset, // some global data
    const Vector3* dN, // Derivative of shape function
    Vector3* p, int* topology, // mesh
    scalar* V, Matrix3x3* JX_inv // element data (Volume * Weight, Inverse of initial jacobian)
    ) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x; // num of element
    if (eid >= n) return;
    const int vid = offset + eid * nb_vert_elem; // first vertice id in topology
    const int qid = eid * nb_quadrature;
    printf("vid = %d   eid = %d   offset = %d", eid, vid, offset);
}

__global__ void kernel_velocity_update(const int n, const float dt, Vector3* p, Vector3* prev_p, Vector3* v) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = (p[i] - prev_p[i]) / dt;
}


__global__ void kernel_step_solver(const int n, const float dt, const Vector3 g, Vector3* p, Vector3* v, Vector3* f, float* w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] += (g + f[i] * w[i])*dt;
    p[i] += v[i] * dt;
    f[i] *= 0;
}

__global__ void kernel_constraint_plane(const int n, const Vector3 origin, const Vector3 normal, Vector3* p, Vector3* v, Vector3* f) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Vector3 d = p[i] - origin;
    scalar s = dot(d, normal);
    if(s > 0 ) return;
    p[i] -= normal * s;
    v[i] = -v[i] * 0.99f;
    f[i] = {};
}


void GPU_PBD_FEM::step(const scalar dt) const {
    kernel_step_solver<<<(cb_position->nb+255)/256, 256>>>(cb_position->nb, dt, Dynamic::gravity(), cb_position->buffer, cb_velocity->buffer, cb_forces->buffer, cb_weights->buffer);
    for(int i = 0; i < nb_color; ++i) {
        //kernel_constraint_solve<<<(cb_position->nb+255)/256, 256>>>(cb_position->nb, );
    }
    kernel_constraint_plane<<<(cb_position->nb+255)/256, 256>>>(cb_position->nb, -Unit3D::up() * 0.1f, Unit3D::up(), cb_position->buffer, cb_velocity->buffer, cb_forces->buffer);

}
