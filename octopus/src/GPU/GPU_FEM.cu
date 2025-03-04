#include "Core/Base.h"
#include "GPU/GPU_FEM.h"
#include "GPU/GPU_FEM_Material.h"
#include<vector>
#include <Manager/Debug.h>

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

std::vector<scalar> GPU_FEM::get_stress(const GPU_ParticleSystem* ps) const
{
    kernel_compute_stress<<<(d_fem->nb_element + 31) / 32, 32>>>(
        *d_material, ps->get_parameters(), get_fem_parameters(), cb_elem_data->buffer
    );

    std::vector<scalar> stress(d_fem->nb_element);
    cb_elem_data->get_data(stress);
    return stress;
}

std::vector<scalar> GPU_FEM::get_volume(const GPU_ParticleSystem* ps) const
{
    kernel_compute_volume<<<(d_fem->nb_element + 31) / 32, 32>>>(
        ps->get_parameters(), get_fem_parameters(), cb_elem_data->buffer
    );
    std::vector<scalar> volumes(d_fem->nb_element);
    cb_elem_data->get_data(volumes);
    return volumes;
}

std::vector<scalar> GPU_FEM::get_volume_diff(const GPU_ParticleSystem* ps) const
{
    kernel_compute_volume_diff<<<(d_fem->nb_element + 31) / 32, 32>>>(
        ps->get_parameters(), get_fem_parameters(), cb_elem_data->buffer
    );
    std::vector<scalar> volumes_diff(d_fem->nb_element);
    cb_elem_data->get_data(volumes_diff);
    return volumes_diff;
}


GPU_FEM::GPU_FEM(const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                 const scalar young, const scalar poisson, const Material material) : d_material(nullptr), d_fem(nullptr)
{
    std::cout << "GPU FEM : NB ELEMENT = " << topology.size() / elem_nb_vertices(element) << std::endl;
    d_thread = new Thread_Data();
    d_material = new Material_Data();
    d_material->material = material;
    d_material->lambda = young * poisson / ((1.f + poisson) * (1.f - 2.f * poisson));
    d_material->mu = young / (2.f * (1.f + poisson));

    cb_elem_data = new Cuda_Buffer(std::vector<scalar>(static_cast<int>(topology.size()) / elem_nb_vertices(element)));
    // rebuild constant for FEM simulation
    d_fem = GPU_FEM::build_fem_const(element, geometry, topology);

}

std::vector<Vector3> GPU_FEM::get_forces(const GPU_ParticleSystem *ps, scalar dt) const {
    return std::vector(ps->nb_particles(), Vector3(0));
}


GPU_FEM_Data* GPU_FEM::build_fem_const(const Element& element, const Mesh::Geometry &geometry, const Mesh::Topology& topology) {

    const FEM_Shape* shape = get_fem_shape(element);
    const int elem_nb_vert = elem_nb_vertices(element);
    const int nb_element = static_cast<int>(topology.size()) / elem_nb_vert;
    const int nb_quadrature = static_cast<int>(shape->weights.size());;

    std::vector<Vector3> dN;
    std::vector<Matrix3x3> JX_inv(nb_quadrature * nb_element);
    std::vector<scalar> V(nb_quadrature * nb_element);
    for (int i = 0; i < nb_quadrature; i++)
        dN.insert(dN.end(), shape->dN[i].begin(), shape->dN[i].end());

    for (int i = 0; i < nb_element; i++) {
        const int id = i * elem_nb_vert;
        const int eid = i * nb_quadrature;
        for (int j = 0; j < nb_quadrature; ++j) {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < elem_nb_vert; ++k) {
                J += glm::outerProduct(geometry[topology[id + k]], dN[j * elem_nb_vert + k]);
            }
            V[eid + j] = abs(glm::determinant(J)) * shape->weights[j];
            JX_inv[eid + j] = glm::inverse(J);
        }
    }

    GPU_FEM_Data* data_fem = new GPU_FEM_Data();
    data_fem->elem_nb_vert = elem_nb_vertices(element);
    data_fem->nb_quadrature = nb_quadrature;
    data_fem->nb_element = nb_element;
    data_fem->cb_weights = new Cuda_Buffer(shape->weights);
    data_fem->cb_topology = new Cuda_Buffer(topology);
    data_fem->cb_dN = new Cuda_Buffer(dN);
    data_fem->cb_V = new Cuda_Buffer(V);
    data_fem->cb_JX_inv = new Cuda_Buffer(JX_inv);

    delete shape;
    return data_fem;
}


