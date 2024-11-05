#include "GPU/GPU_PBD_FEM_Coupled.h"
GPU_PBD_FEM_Coupled::GPU_PBD_FEM_Coupled(const Element element, const Mesh::Geometry &geometry, const Mesh::Topology &topology, const scalar young, const scalar poisson)
        : GPU_PBD_FEM(element, geometry, topology, young, poisson) { }
