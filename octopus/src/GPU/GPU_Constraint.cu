#include "Core/Base.h"
#include "GPU/GPU_Constraint.h"
#include<vector>
#include <GPU/GPU_ParticleSystem.h>
#include <Manager/Debug.h>

__global__ void kernel_constraint_fix(const int n, const Vector3 com,const Vector3 offset, const Matrix3x3 rot, const int* ids,
    GPU_ParticleSystem_Parameters ps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int id = ids[i];
    const Vector3 target = offset + com + rot * (ps.init_p[id] - com);
    ps.p[id] = target;
    ps.v[id] = Vector3(0, 0, 0);
    ps.f[id] = Vector3(0, 0, 0);
    ps.mask[id] = 0;
}


__global__ void kernel_constraint_crush(GPU_ParticleSystem_Parameters ps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    if(ps.mask[i] == 0) return;
    //const Vector3 p_init = com + rot * (ps.p[i] - com);
    //ps.p[i] = p_init - glm::dot(p_init - origin, normal) * normal + offset;
    ps.p[i].y = 0;
    ps.last_p[i] = ps.p[i];
    ps.v[i] = Vector3(0, 0, 0);
    ps.f[i] = Vector3(0, 0, 0);
}

__device__ float fracf(float x)
{
    return x - floorf(x);
}

__device__ scalar scalar_to_scalar_random(scalar v) {
    return fracf(sin(v*(91.3458f)) * 47453.5453f);
}
__device__ scalar vec2_to_scalar_random(Vector2 v) {
    return fracf(sin(dot(v ,Vector2(12.9898f,78.233f))) * 43758.5453f);
}

__device__ scalar vec3_to_scalar_random(Vector3 v) {
    return vec2_to_scalar_random(Vector2(v.x,v.y)+Vector2(scalar_to_scalar_random(v.z)));
}

__global__ void kernel_constraint_random_sphere(Vector3 c, scalar radius, GPU_ParticleSystem_Parameters ps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    if(ps.mask[i] == 0) return;

    const scalar rx = vec3_to_scalar_random(ps.init_p[i]) * 2.f - 1.;
    const scalar ry = scalar_to_scalar_random(rx * i) * 2.f - 1.;
    const scalar rz = scalar_to_scalar_random(rx * ry) * 2.f - 1.;
    const scalar r = scalar_to_scalar_random(rz*rx*ry);

    const Vector3 d = glm::normalize(Vector3(rx,ry,rz));
    ps.p[i] = c + d * radius * r;
    ps.last_p[i] = ps.p[i];
    ps.v[i] = Vector3(0, 0, 0);
    ps.f[i] = Vector3(0, 0, 0);
}


__global__ void kernel_constraint_box_limit(const Vector3 pmin, const Vector3 pmax, GPU_ParticleSystem_Parameters ps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    if(ps.mask[i] == 0) return;

    Vector3 p = ps.p[i];
    p.x = glm::clamp(p.x, pmin.x, pmax.x);
    p.y = glm::clamp(p.y, pmin.y, pmax.y);
    p.z = glm::clamp(p.z, pmin.z, pmax.z);
    ps.p[i] = p;
}




GPU_Fix_Constraint::GPU_Fix_Constraint(const Mesh::Geometry& positions, Area* area) {
    std::vector<int> ids;
    com = Vector3(0,0,0);
    int count = 0;
    for(size_t i = 0; i < positions.size(); ++i) {
        if(area->inside(positions[i])) {
            count++;
            com += positions[i];
            ids.push_back(i);
        }
    }
    com /= count;
    cb_ids = new Cuda_Buffer<int>(ids);
}

void GPU_Fix_Constraint::step(GPU_ParticleSystem* ps, const scalar dt)
{
    kernel_constraint_fix<<<(cb_ids->nb + 31) / 32, 32>>>(
        cb_ids->nb, com, axis.position(), axis.rotation(),
        cb_ids->buffer, ps->get_parameters());
}

void GPU_Crush::step(GPU_ParticleSystem *ps, scalar dt) {
    kernel_constraint_crush<<<(ps->nb_particles() + 31) / 32, 32>>>(ps->get_parameters());
}


void GPU_RandomSphere::step(GPU_ParticleSystem *ps, scalar dt) {
    kernel_constraint_random_sphere<<<(ps->nb_particles() + 31) / 32, 32>>>(center, radius, ps->get_parameters());
}

void GPU_Box_Limit::step(GPU_ParticleSystem *ps, scalar dt) {
    kernel_constraint_box_limit<<<(ps->nb_particles() + 31) / 32, 32>>>(pmin, pmax, ps->get_parameters());
}