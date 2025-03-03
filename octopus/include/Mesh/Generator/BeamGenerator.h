#pragma once
#include "Mesh/Generator/MeshGenerator.h"
#include "Mesh/Converter/MeshConverter.h"

struct BeamMeshGenerator : public MeshGenerator {
    BeamMeshGenerator(const Vector3I &subdivisions, const Vector3 &sizes)
        : _subdivisions(subdivisions + Vector3I(1)), _sizes(sizes) {
        _x_step = _sizes.x / scalar(_subdivisions.x - 1);
        _y_step = _sizes.y / scalar(_subdivisions.y - 1);
        _z_step = _sizes.z / scalar(_subdivisions.z - 1);
    }


    Mesh *build() override;

    void build_geometry_grid(Mesh::Geometry &geometry) const;

    void get_cell_vertices_ids(int x, int y, int z, int *ids) const;

    [[nodiscard]] int icoord_to_id(int x, int y, int z) const;

    virtual void add_geometry_at_cell(int x, int y, int z, Mesh::Geometry &geometry) {
    }

    virtual void build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) = 0;

protected:
    Vector3I _subdivisions;
    Vector3 _sizes;
    scalar _x_step, _y_step, _z_step;
};

class HexaBeamGenerator : public BeamMeshGenerator {
public:
    HexaBeamGenerator(const Vector3I &_subdivisions, const Vector3 &_sizes)
        : BeamMeshGenerator(_subdivisions, _sizes) {
    }

    void build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) override;
};

class PrismBeamGenerator : public BeamMeshGenerator {
public:
    PrismBeamGenerator(const Vector3I &_subdivisions, const Vector3 &_sizes)
        : BeamMeshGenerator(_subdivisions, _sizes) {
    }

    void build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) override;
};

class PyramidBeamGenerator : public BeamMeshGenerator {
public:
    int _mid_id;

    PyramidBeamGenerator(const Vector3I &_subdivisions, const Vector3 &_sizes)
        : BeamMeshGenerator(_subdivisions, _sizes), _mid_id(0) {
    }

    void add_geometry_at_cell(int x, int y, int z, Mesh::Geometry &geometry) override;

    void build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) override;
};


class TetraBeamGenerator : public BeamMeshGenerator {
public:
    TetraBeamGenerator(const Vector3I &_subdivisions, const Vector3 &_sizes)
        : BeamMeshGenerator(_subdivisions, _sizes) {
    }

    void build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) override;
};


void subdive_tetra(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies);

void tetra4_to_tetra10(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies);

void tetra4_to_tetra20(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies);

void hexa_to_hexa27(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies);

// convert data in an intial mesh to a target mesh
struct MeshMap {
    Element s_elem; // what is the initial element type
    Element t_elem; // what is the target element type
    Mesh::Geometry ref_geometry; // vertices position in reference element
    Mesh::Topology elem_topo; // topology of linear elem
    std::vector<int> v_elem; // in which element the vertices is valid


    MeshMap(const Element &_s_elem, const Element &_t_elem,
            const Mesh::Geometry &_ref_geometry,
            const Mesh::Topology &_elem_topo,
            const std::vector<int> &_v_elem)
        : s_elem(_s_elem), t_elem(_t_elem), ref_geometry(_ref_geometry), elem_topo(_elem_topo), v_elem(_v_elem) {
    }

    template<typename T>
    std::vector<T> convert(Mesh *mesh, const std::vector<T> &vals);

    void apply_to_mesh(Mesh *mesh);
};


void tetra_refine(MeshMap *map, Mesh::Geometry &ref_tetra_geometry, Mesh::Topology &ref_tetra_edges,
                  std::vector<int> &t_ids);


/// Create a mapping of a Tetra mesh into linear Tetra (that can be refined)
MeshMap *tetra_to_linear(Mesh *mesh, Element elem, int subdivision);
MeshMap *hexa_to_linear(Mesh *mesh, Element elem, int subdivision);