#include <vector>
#include "Dynamic/FEM/FEM_Shape.h"
#include "Mesh/Elements.h"
#include <iostream>

void FEM_Shape::build() {
    const std::vector<scalar> coords = get_quadrature_coordinates();
    weights = get_weights();
    dN.resize(weights.size());
    for (int i = 0; i < weights.size(); ++i) {
        dN[i] = build_shape_derivatives(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
    }
}

std::vector<Vector3> FEM_Shape::convert_dN_to_vector3(scalar* dN) const {
    std::vector<Vector3> dN_v3(nb);
    for (int i = 0; i < nb; ++i) {
        dN_v3[i].x = dN[i];
        dN_v3[i].y = dN[i + nb];
        dN_v3[i].z = dN[i + nb * 2];
    }
    return dN_v3;
}


FEM_Shape *get_fem_shape(Element type) {
    switch (type) {
        case Tetra: return new Tetra_4();
        case Pyramid: return new Pyramid_5();
        case Prism: return new Prism_6();
        case Hexa: return new Hexa_8();
        case Tetra10: return new Tetra_10();
        case Tetra20: return new Tetra_20();
        default: std::cout << "build_element : element not found " << type << std::endl;
            return nullptr;
    }
}
std::vector<scalar> compute_fem_mass(const Element& elem, const Mesh::Geometry& geometry, const Mesh::Topology& topology, const scalar density)
{
    const int nb_vert_elem = elem_nb_vertices(elem);
    const scalar v_density = density / static_cast<scalar>(nb_vert_elem);
    FEM_Shape* shape = get_fem_shape(elem); shape->build();

    std::vector<scalar> mass(geometry.size());
    for(int i = 0; i < topology.size(); i+= nb_vert_elem) {
        scalar V = 0.f;
        for(int q = 0; q < shape->weights.size(); ++q) {
            Matrix3x3 J = Matrix::Zero3x3();
            for(int j = 0; j < nb_vert_elem; j++) {
                const int vid = topology[i + j];
                J+= glm::outerProduct(geometry[vid], shape->dN[q][j]);
            }
            V += abs(glm::determinant(J)) * shape->weights[q];
        }

        for(int j = 0; j < nb_vert_elem; j++) {
            const int vid = topology[i + j];
            mass[vid] += v_density * V;
        }
    }
    delete shape;
    return mass;
}