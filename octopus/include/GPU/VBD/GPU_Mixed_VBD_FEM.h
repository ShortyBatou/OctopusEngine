#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Mesh/Mesh.h>
#include "Core/Base.h"
#include "Mesh/Elements.h"
#include "GPU/VBD/GPU_VBD_FEM.h"

struct GPU_Mixed_VBD_FEM final : public GPU_VBD_FEM
{
    GPU_Mixed_VBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry, const Material& material,
        const scalar& young, const scalar& poisson, const scalar& damping);
    void explicit_step(const GPU_ParticleSystem* ps, Cuda_Buffer<scalar>* w_max, scalar dt) const;
    Thread_Data* d_exp_thread;

    Cuda_Buffer<Vector3>* p_forces;

    ~GPU_Mixed_VBD_FEM() override
    {
        delete d_exp_thread;
    }
};
