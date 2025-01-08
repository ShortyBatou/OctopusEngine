#include "GPU/GPU_FEM.h"
#include <GPU/CUMatrix.h>
#include "GPU/GPU_FEM_Material.h"

// fem global function
__device__ Matrix3x3 compute_transform(const int nb_vert_elem, const Vector3* pos, const int* topology,
                                       const Vector3* dN)
{
    // Compute transform (reference => scene)
    Matrix3x3 Jx = Matrix3x3(0.f);
    for (int j = 0; j < nb_vert_elem; ++j)
    {
        Jx += glm::outerProduct(pos[topology[j]], dN[j]);
    }
    return Jx;
}

__global__ void kernel_constraint_plane(const Vector3 origin, const Vector3 normal, const Vector3 com,
                                        const Vector3 offset, const Matrix3x3 rot,
                                        GPU_ParticleSystem_Parameters ps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    scalar s = dot(ps.init_p[i] - origin, normal);
    if (s > 0)
    {
        const Vector3 target = offset + com + rot * (ps.init_p[i] - com);
        ps.p[i] = target;
        ps.v[i] = Vector3(0, 0, 0);
        ps.f[i] = Vector3(0, 0, 0);
        ps.mask[i] = 0;
    }
}

__global__ void kernel_constraint_plane_crush(const Vector3 origin, const Vector3 normal, const Vector3 com,
                                        const Vector3 offset, const Matrix3x3 rot,
                                        GPU_ParticleSystem_Parameters ps)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.nb_particles) return;
    const Vector3 p_init = com + rot * (ps.init_p[i] - com);
    ps.p[i].y = 0; //p_init - glm::dot(p_init - origin, normal) * normal + offset;
    ps.last_p[i] = ps.p[i];
    ps.v[i] = Vector3(0, 0, 0);
    ps.f[i] = Vector3(0, 0, 0);
}

__global__ void kernel_compute_stress(
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    scalar* stress)
{
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= fem.nb_element) return;
    const int qid = eid * fem.nb_quadrature;
    scalar s = 0;
    const int* topo = fem.topology + eid * fem.elem_nb_vert;
    // offset the pointer at the start of the element's topology
    Matrix3x3 Jx(0.f);
    for (int i = 0; i < fem.nb_quadrature; ++i)
    {
        for (int j = 0; j < fem.elem_nb_vert; ++j)
        {
            Jx += glm::outerProduct(ps.p[topo[j]], fem.dN[i * fem.elem_nb_vert + j]);
        }
        Matrix3x3 F = Jx * fem.JX_inv[qid + i];
        Matrix3x3 P = eval_pk1_stress(mt.material, mt.lambda, mt.mu, F);
        s += von_mises_stress(pk1_to_cauchy_stress(F, P)) * fem.V[qid + i];
    }
    stress[eid] = s;
}

__global__ void kernel_compute_volume(
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    scalar* volumes)
{
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= fem.nb_element) return;
    scalar volume = 0;
    const int* topo = fem.topology + eid * fem.elem_nb_vert;
    // offset the pointer at the start of the element's topology
    Matrix3x3 Jx(0.f);
    for (int i = 0; i < fem.nb_quadrature; ++i)
    {
        for (int j = 0; j < fem.elem_nb_vert; ++j)
        {
            Jx += glm::outerProduct(ps.p[topo[j]], fem.dN[i * fem.elem_nb_vert + j]);
        }
        volume += fabsf(glm::determinant(Jx)) * fem.weights[i];
    }
    volumes[eid] = volume;
}

__global__ void kernel_compute_volume_diff(
    GPU_ParticleSystem_Parameters ps,
    GPU_FEM_Pameters fem,
    scalar* diff_volumes)
{
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= fem.nb_element) return;
    scalar diff_volume = 0;
    const int qid = eid * fem.nb_quadrature;
    const int* topo = fem.topology + eid * fem.elem_nb_vert;
    Matrix3x3 Jx(0.f);
    for (int i = 0; i < fem.nb_quadrature; ++i)
    {
        for (int j = 0; j < fem.elem_nb_vert; ++j)
        {
            Jx += glm::outerProduct(ps.p[topo[j]], fem.dN[i * fem.elem_nb_vert + j]);
        }
        diff_volume += fabsf(glm::determinant(Jx)) * fem.weights[i] - fem.V[qid + i];
    }
    diff_volumes[eid] = diff_volume;
}

std::vector<scalar> GPU_FEM::get_stress(GPU_ParticleSystem* ps) const
{
    kernel_compute_stress<<<(d_fem->nb_element + 31) / 32, 32>>>(
        *d_material, ps->get_parameters(), get_fem_parameters(), cb_elem_data->buffer
    );

    std::vector<scalar> stress(d_fem->nb_element);
    cb_elem_data->get_data(stress);
    return stress;
}

std::vector<scalar> GPU_FEM::get_volume(GPU_ParticleSystem* ps) const
{
    kernel_compute_volume<<<(d_fem->nb_element + 31) / 32, 32>>>(
        ps->get_parameters(), get_fem_parameters(), cb_elem_data->buffer
    );
    std::vector<scalar> volumes(d_fem->nb_element);
    cb_elem_data->get_data(volumes);
    return volumes;
}

std::vector<scalar> GPU_FEM::get_volume_diff(GPU_ParticleSystem* ps) const
{
    kernel_compute_volume_diff<<<(d_fem->nb_element + 31) / 32, 32>>>(
        ps->get_parameters(), get_fem_parameters(), cb_elem_data->buffer
    );
    std::vector<scalar> volumes_diff(d_fem->nb_element);
    cb_elem_data->get_data(volumes_diff);
    return volumes_diff;
}


void GPU_Plane_Fix::step(GPU_ParticleSystem* ps, const scalar dt)
{

    if(all) kernel_constraint_plane_crush<<<(ps->nb_particles() + 255) / 256, 256>>>(origin, normal, com, offset, rot, ps->get_parameters());
    else kernel_constraint_plane<<<(ps->nb_particles() + 255) / 256, 256>>>(origin, normal, com, offset, rot, ps->get_parameters());

}
