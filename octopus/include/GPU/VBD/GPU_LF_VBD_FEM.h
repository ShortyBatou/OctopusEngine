#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Mesh/Mesh.h>
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/VBD/GPU_VBD_FEM.h"

struct GPU_LF_VBD_FEM final : public GPU_VBD_FEM
{
    GPU_LF_VBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry, const Material& material,
        const scalar& young, const scalar& poisson, const scalar& damping);

    void start(GPU_ParticleSystem* ps, scalar dt) override;
    void step(GPU_ParticleSystem* ps, scalar dt) override;
    Cuda_Buffer<scalar>* l;
    Cuda_Buffer<scalar>* Vi;
    ~GPU_LF_VBD_FEM() override = default;
};
