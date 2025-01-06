#include "GPU/VBD/GPU_LF_VBD_FEM.h"
#include <set>

GPU_LF_VBD_FEM::GPU_LF_VBD_FEM(const Element &element, const Mesh::Topology &topology, const Mesh::Geometry &geometry,
                         const Material& material, const scalar &young, const scalar &poisson, const scalar& damping) :
    GPU_VBD_FEM(element, topology, geometry, material, young, poisson, damping)
{
    l = new Cuda_Buffer<scalar>(std::vector<scalar>(geometry.size(), 0.f));
}