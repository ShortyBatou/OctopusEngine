#pragma once
#include "Core/Base.h"
#include "GPU/GPU_ParticleSystem.h"
#include "GPU/GPU_FEM.h"

GPU_Plane_Fix::GPU_Plane_Fix(const Mesh::Geometry& positions, const Vector3& o, const Vector3& n)
: com(Unit3D::Zero()), offset(Unit3D::Zero()), origin(o), normal(n), rot(Matrix::Identity3x3()) {
    int count = 0;
    for(auto& p : positions) {
        const Vector3 dir = p - origin;
        const scalar d = dot(dir, normal);
        if(d > 0.f) {
            count++;
            com += p;
        }
    }
    com /= count;
    all = false;
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