#include "GPU/GPU_FEM.h"
#include <GPU/CUMatrix.h>

// fem global function
__device__ Matrix3x3 compute_transform(int nb_vert_elem, Vector3 *pos, int *topology, Vector3 *dN) {
    // Compute transform (reference => scene)
    Matrix3x3 Jx = Matrix3x3(0.f);
    for (int j = 0; j < nb_vert_elem; ++j) {
        Jx += glm::outerProduct(pos[topology[j]], dN[j]);
    }
    return Jx;
}

__global__ void kernel_constraint_plane(const int n, const Vector3 origin, const Vector3 normal, const Vector3 com, const Vector3 offset, const Matrix3x3 rot, Vector3 *p, Vector3 *p_init, int* mask) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = dot(p_init[i] - origin, normal);
    if (s > 0) {
        const Vector3 target = offset + com + rot * (p_init[i] - com);
        p[i] = target;
        mask[i] = 0;
    }
}

void GPU_Plane_Fix::step(const GPU_ParticleSystem *ps, const scalar dt) {
    kernel_constraint_plane<<<(ps->nb_particles() + 255) / 256, 256>>>(
        ps->nb_particles(), origin, normal, com, offset, rot,
        ps->buffer_position(), ps->buffer_init_position(), ps->buffer_mask());
}