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
}


GPU_FEM::GPU_FEM(const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, // mesh
                 const scalar young, const scalar poisson, const Material material)
{
    _material = material;
    shape = get_fem_shape(element);

    lambda = young * poisson / ((1.f + poisson) * (1.f - 2.f * poisson));
    mu = young / (2.f * (1.f + poisson));

    elem_nb_vert = elem_nb_vertices(element);
    nb_quadrature = static_cast<int>(shape->weights.size());
    nb_element = static_cast<int>(topology.size()) / elem_nb_vert;
    // rebuild constant for FEM simulation
    GPU_FEM::build_fem_const(geometry, topology);
}

void GPU_FEM::build_fem_const(const Mesh::Geometry &geometry, const Mesh::Topology& topology) {
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

    cb_dN = new Cuda_Buffer(dN);
    cb_V = new Cuda_Buffer(V);
    cb_JX_inv = new Cuda_Buffer(JX_inv);
}