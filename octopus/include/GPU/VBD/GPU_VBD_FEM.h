#pragma once
#include <Core/Base.h>
#include <Mesh/Elements.h>
#include <Mesh/Mesh.h>
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>

#include <GPU/Cuda_Buffer.h>
#include <GPU/GPU_Dynamic.h>
#include <GPU/GPU_FEM.h>


struct GPU_Owners_Parameters
{
    int* offset; // offset to the first neighbors for each vertice
    int* nb; // number of neighbors for each vertices
    int* eids; // element's id that own a vertices
    int* ref_vid; // local vertice id in each neighbors element
};

struct GPU_Owners_Data
{
    Cuda_Buffer<int>* cb_offset; // offset to the first owners for each vertice
    Cuda_Buffer<int>* cb_nb; // number of neighbors for each vertices
    Cuda_Buffer<int>* cb_eids; // element's id that own a vertices
    Cuda_Buffer<int>* cb_ref_vid; // local vertice id in each neighbors element

    ~GPU_Owners_Data()
    {
        delete cb_offset;
        delete cb_nb;
        delete cb_eids;
        delete cb_ref_vid;
    }
};


struct GPU_VBD_FEM : GPU_FEM
{
    GPU_VBD_FEM(const Element& element, const Mesh::Topology& topology, const Mesh::Geometry& geometry,
                const Material& material,
                const scalar& young, const scalar& poisson, const scalar& damping);

    void step(GPU_ParticleSystem* ps, scalar dt) override;

    //std::vector<Vector3> get_residual(const GPU_ParticleSystem *ps, scalar dt) const override;

    void build_graph_color(
        const Mesh::Topology& topology,
        int nb_vertices,
        std::vector<int>& colors,
        std::vector<std::vector<int>>& e_neighbors,
        std::vector<std::vector<int>>& ref_id) const;


    void GPU_VBD_FEM::sort_by_color(int nb_vertices, const std::vector<int>& colors, const std::vector<std::vector<int>>& e_owners,
                                            const std::vector<std::vector<int>>& e_ref_id) const;

    [[nodiscard]] GPU_Owners_Parameters get_owners_parameters() const
    {
        GPU_Owners_Parameters params{};
        params.offset = d_owners->cb_offset->buffer;
        params.nb = d_owners->cb_nb->buffer;
        params.eids = d_owners->cb_eids->buffer;
        params.ref_vid = d_owners->cb_ref_vid->buffer;
        return params;
    }

    GPU_Owners_Data* d_owners;
    scalar _damping;
    Cuda_Buffer<Vector3>* r;
    Cuda_Buffer<Vector3>* y; // gets ot from VBD solver
    std::vector<int> _colors; // mesh coloration (used for debug)
};

__global__ void kernel_vbd_solve(
    int n, scalar damping, scalar dt, int offset,const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem, GPU_Owners_Parameters owners
);


__global__ void kernel_vbd_compute_residual(
    int n, scalar damping, scalar dt, int offset,const Vector3* y, Vector3* r,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem, GPU_Owners_Parameters owners
);
