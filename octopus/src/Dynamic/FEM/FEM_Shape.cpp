#include <vector>
#include "Dynamic/FEM/FEM_Shape.h"
#include "Mesh/Elements.h"
#include <iostream>
#include <Dynamic/FEM/FEM_Generic.h>

void FEM_Shape::build()
{
    const std::vector<scalar> coords = get_quadrature_coordinates();
    weights = get_weights();
    dN.resize(weights.size());
    for (int i = 0; i < weights.size(); ++i)
    {
        dN[i] = build_shape_derivatives(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
    }
}

std::vector<Vector3> FEM_Shape::convert_dN_to_vector3(scalar* dN) const
{
    std::vector<Vector3> dN_v3(nb);
    for (int i = 0; i < nb; ++i)
    {
        dN_v3[i].x = dN[i];
        dN_v3[i].y = dN[i + nb];
        dN_v3[i].z = dN[i + nb * 2];
    }
    return dN_v3;
}


FEM_Shape* get_fem_shape(const Element type)
{
    FEM_Shape* fem;
    switch (type)
    {
    case Tetra: fem = new Tetra_4();
        break;
    case Pyramid: fem = new Pyramid_5();
        break;
    case Prism: fem = new Prism_6();
        break;
    case Hexa: fem = new Hexa_8();
        break;
    case Tetra10: fem = new Tetra_10();
        break;
    case Tetra20: fem = new Tetra_20();
        break;
    case Hexa27: fem = new Hexa_27();
        break;
    default: std::cout << "build_element : element not found " << type << std::endl;
        fem = new Tetra_4();
        break;
    }
    return fem;
}

void get_fem_const(const Element& elem, const Mesh::Geometry& geometry, const Mesh::Topology& topology,
                   std::vector<std::vector<Matrix3x3>>& JX_inv, std::vector<std::vector<scalar>>& V)
{
    const FEM_Shape* shape = get_fem_shape(elem);
    const int nb_element = static_cast<int>(topology.size()) / shape->nb;
    const int nb_quadrature = shape->nb_quadratures();

    JX_inv.resize(nb_element);
    V.resize(nb_element);

    for (int i = 0; i < nb_element; i++)
    {
        const int id = i * shape->nb;
        V[i].resize(nb_quadrature);
        JX_inv[i].resize(nb_quadrature);
        for (int j = 0; j < nb_quadrature; ++j)
        {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < shape->nb; ++k)
            {
                J += glm::outerProduct(geometry[topology[id + k]], shape->dN[j][k]);
            }
            V[i][j] = abs(glm::determinant(J)) * shape->weights[j];
            JX_inv[i][j] = glm::inverse(J);
        }
    }
    delete shape;
}

std::vector<scalar> compute_fem_mass(const Element& elem, const Mesh::Geometry& geometry,
                                     const Mesh::Topology& topology, const scalar density, Mass_Distribution distrib)
{
    const int nb_vert_elem = elem_nb_vertices(elem);
    const FEM_Shape* shape = get_fem_shape(elem);

    std::vector<scalar> mass(geometry.size());
    if (distrib == Mass_Distribution::Shape)
    {
        const scalar v_density = density / static_cast<scalar>(nb_vert_elem);
        for (int i = 0; i < topology.size(); i += nb_vert_elem)
        {
            std::vector<int> e_topo(topology.begin() + i, topology.begin() + i + nb_vert_elem);
            const scalar V = FEM_Generic::compute_volume(shape, geometry, e_topo);
            for (int j = 0; j < nb_vert_elem; j++)
            {
                const int vid = topology[i + j];
                mass[vid] += v_density * V;
            }
        }
    }
    else // uniform
    {
        scalar p_mass = 0;
        for (int i = 0; i < topology.size(); i += nb_vert_elem)
        {
            std::vector<int> e_topo(topology.begin() + i, topology.begin() + i + nb_vert_elem);
            p_mass += FEM_Generic::compute_volume(shape, geometry, e_topo);
        }
        p_mass *= density;
        p_mass /= geometry.size();
        for (float& m : mass) m = p_mass;
    }
    delete shape;
    return mass;
}
