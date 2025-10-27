#pragma once
#include <Core/Base.h>
#include <Mesh/Elements.h>
#include <Mesh/Mesh.h>
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>

#include <GPU/Cuda_Buffer.h>
#include <GPU/GPU_Dynamic.h>
#include <GPU/GPU_FEM.h>
#include "GPU/VBD/GPU_VBD_FEM.h"
#include <Tools/Graph.h>

struct GPU_AVBD_FEM final : GPU_VBD_FEM
{
    GPU_AVBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry,
        const Material& material, const scalar& young, const scalar& poisson, const scalar& damping) :
    GPU_VBD_FEM(element, topology, geometry, material,young, poisson, damping) {}

    void step(GPU_ParticleSystem* ps, scalar dt) override;

    ~GPU_AVBD_FEM() override {
        delete lambda;
        delete k;
    }

    scalar gamma, alpha, beta;
    Cuda_Buffer<scalar>* lambda; // for each constraint
    Cuda_Buffer<scalar>* k; // for each constraint
};
