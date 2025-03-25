#pragma once
#include <Core/Base.h>
#include <Mesh/Elements.h>
#include <Mesh/Mesh.h>
#include <Dynamic/FEM/ContinuousMaterial.h>
#include <Dynamic/FEM/FEM_Shape.h>

#include <GPU/Cuda_Buffer.h>
#include <GPU/GPU_Dynamic.h>
#include <GPU/GPU_FEM.h>
#include <Tools/Graph.h>

enum VBD_Version {
    Base, Threaded_Quadrature, Reduction_Symmetry, Better_Coloration, Block_Merge
};

struct GPU_BLock_Parameters {
    int* sub_block_size;
    int* nb_sub_block;
    int* offset;
};

struct GPU_Block_Data {
    Cuda_Buffer<int>* cb_offset; // offset to the first owners for each vertice
    Cuda_Buffer<int>* cb_sub_block_size; // number of neighbors for each vertices
    Cuda_Buffer<int>* cb_nb_sub_block; // element's id that own a vertices

    ~GPU_Block_Data()
    {
        delete cb_offset;
        delete cb_sub_block_size;
        delete cb_nb_sub_block;
    }
};

struct GPU_Split_Parameters {
    Vector3* position; // N + S
    int* true_id; // N + S

    int* nb_split; // S
    int* off_split; // S
    int* id_split; //
    scalar* weight; //
};

struct GPU_Split_Data {
    Cuda_Buffer<Vector3>* cb_position;
    Cuda_Buffer<int>* cb_true_id;
    Cuda_Buffer<int>* cb_nb_split;
    Cuda_Buffer<int>* cb_off_split;
    Cuda_Buffer<int>* cb_id_split;
    Cuda_Buffer<scalar>* cb_weight;

    ~GPU_Split_Data()
    {
        delete cb_position;
        delete cb_true_id;
        delete cb_nb_split;
        delete cb_off_split;
        delete cb_id_split;
        delete cb_weight;
    }
};

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
                const scalar& young, const scalar& poisson, const scalar& damping,
                const VBD_Version& v = VBD_Version::Better_Coloration);

    void step(GPU_ParticleSystem* ps, scalar dt) override;

    std::vector<Vector3> get_forces(const GPU_ParticleSystem *ps, scalar dt) const override;

    Coloration build_graph_color(Element element, const Mesh::Topology& topology);

    void create_buffers(Element element,
        const Mesh::Topology& topology,
        Coloration& coloration,
        std::vector<std::vector<int>>& e_owners,
        std::vector<std::vector<int>>& e_ref_id
    );

    void build_owner_data(int nb_vertices, const Mesh::Topology &topology, std::vector<std::vector<int>>& e_neighbors, std::vector<std::vector<int>>& e_ref_id) const;
    [[nodiscard]] GPU_Owners_Parameters get_owners_parameters() const
    {
        GPU_Owners_Parameters params{};
        params.offset = d_owners->cb_offset->buffer;
        params.nb = d_owners->cb_nb->buffer;
        params.eids = d_owners->cb_eids->buffer;
        params.ref_vid = d_owners->cb_ref_vid->buffer;
        return params;
    }

    [[nodiscard]] GPU_BLock_Parameters get_block_parameters() const
    {
        GPU_BLock_Parameters params{};
        params.offset = d_blocks->cb_offset->buffer;
        params.sub_block_size = d_blocks->cb_sub_block_size->buffer;
        params.nb_sub_block = d_blocks->cb_nb_sub_block->buffer;
        return params;
    }

    [[nodiscard]] GPU_Split_Parameters get_Split_parameters() const
    {
        GPU_Split_Parameters params{};
        params.position = d_splits->cb_position->buffer;
        params.true_id = d_splits->cb_true_id->buffer;
        params.nb_split = d_splits->cb_nb_split->buffer;
        params.off_split = d_splits->cb_off_split->buffer;
        params.id_split = d_splits->cb_id_split->buffer;
        params.weight = d_splits->cb_weight->buffer;
        return params;
    }

    ~GPU_VBD_FEM() override {
        delete d_owners;
        delete d_blocks;
        delete r;
    }

    GPU_Owners_Data* d_owners;
    GPU_Split_Data* d_splits;
    GPU_Block_Data* d_blocks;

    std::vector<int> _t_color;
    int _t_nb_color;
    std::map<int, int> _t_conflict;

    VBD_Version version;
    Graph* p_graph;
    Graph* d_graph;
    scalar damping;
    Cuda_Buffer<Vector3>* r;
    Cuda_Buffer<Vector3>* y; // gets ot from VBD solve
    std::vector<int> _colors; // mesh coloration (used for debug)
};

__global__ void kernel_vbd_solve_v1(
    int n, scalar damping, scalar dt, int offset,const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem, GPU_Owners_Parameters owners
);

__global__ void kernel_vbd_solve_v2(
    int n, scalar damping, scalar dt, int offset,const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem, GPU_Owners_Parameters owners
);

__global__ void kernel_vbd_solve_v3(
    int n, scalar damping, scalar dt, int offset,const Vector3* y,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem, GPU_Owners_Parameters owners
);

__global__ void kernel_vbd_compute_residual(
    int n, scalar damping, scalar dt, int offset,const Vector3* y, Vector3* r,
    Material_Data mt,
    GPU_ParticleSystem_Parameters ps, GPU_FEM_Pameters fem, GPU_Owners_Parameters owners
);
