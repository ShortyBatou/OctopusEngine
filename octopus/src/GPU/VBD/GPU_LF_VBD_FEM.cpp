#include "GPU/VBD/GPU_LF_VBD_FEM.h"
#include <set>

GPU_LF_VBD_FEM::GPU_LF_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping) :
    GPU_VBD_FEM(element, topology, geometry, material, young, poisson, damping)
{
    l = new Cuda_Buffer<scalar>(std::vector<scalar>(geometry.size(), 0.f));

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