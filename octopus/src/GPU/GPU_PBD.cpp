#include "GPU/GPU_PBD.h"

GPU_PBD_FEM::GPU_PBD_FEM(Element element, const std::vector<Vector3>& geometry, const std::vector<int>& topology, const std::vector<int>& offsets, const float density)
{
    FEM_Shape* shape = get_fem_shape(element);
    shape->build();

    elem_nb_vert = elem_nb_vertices(element);
    nb_verts = static_cast<int>(geometry.size());
    nb_quadrature = static_cast<int>(shape->weights.size());
    c_offsets = offsets;

    const int nb_elem = static_cast<int>(topology.size()) / elem_nb_vert;

    for(int i = 0; i < offsets.size(); ++i) {
        const int nb = (i < offsets.size()-1 ? offsets[i+1] : topology.size()) - offsets[i];
        c_nb_elem.push_back(nb / elem_nb_vert);
    }

    std::vector<scalar> mass(geometry.size());
    std::vector<scalar> inv_mass(mass.size());
    std::vector<Vector3> dN;
    std::vector<Matrix3x3> JX_inv(nb_quadrature * nb_elem);
    std::vector<scalar> V(nb_quadrature * nb_elem);
    for (int i = 0; i < nb_quadrature; i++)
    {
        dN.insert(dN.end(), shape->dN[i].begin(), shape->dN[i].end());
    }

    std::cout << "NB ELEMENT : " << nb_elem << std::endl;
    for (int i = 0; i < nb_elem; i++)
    {
        scalar V_sum = 0;
        const int id = i * elem_nb_vert;
        const int eid = i * nb_quadrature;
        for (int j = 0; j < nb_quadrature; ++j)
        {
            Matrix3x3 J = Matrix::Zero3x3();
            for (int k = 0; k < elem_nb_vert; ++k)
            {
                J += glm::outerProduct(geometry[topology[id + k]], dN[j * elem_nb_vert + k]);
            }
            V[eid + j] = abs(glm::determinant(J)) * shape->weights[j];
            JX_inv[eid + j] = glm::inverse(J);
            V_sum += V[eid + j];
        }

        for (int j = 0; j < elem_nb_vert; ++j)
        {
            mass[topology[id + j]] += density * V_sum / static_cast<scalar>(elem_nb_vert);
        }
    }

    for (int i = 0; i < mass.size(); ++i)
    {
        inv_mass[i] = 1.f / mass[i];
    }

    cb_position = new Cuda_Buffer(geometry);
    cb_prev_position = new Cuda_Buffer(geometry);
    cb_init_position = new Cuda_Buffer(geometry);
    cb_velocity = new Cuda_Buffer(std::vector(nb_verts, Unit3D::Zero()));
    cb_forces = new Cuda_Buffer(std::vector(nb_verts, Unit3D::Zero()));
    cb_mass = new Cuda_Buffer(mass);
    cb_inv_mass = new Cuda_Buffer(inv_mass);

    cb_topology = new Cuda_Buffer(topology);
    cb_weights = new Cuda_Buffer<scalar>(shape->get_weights());
    cb_dN = new Cuda_Buffer(dN);
    cb_V = new Cuda_Buffer(V);
    cb_JX_inv = new Cuda_Buffer(JX_inv);
}