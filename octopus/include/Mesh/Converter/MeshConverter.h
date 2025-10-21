#pragma once
#include "Mesh/Mesh.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Manager/Debug.h"

struct MeshConverter {
    MeshConverter() : _shape(nullptr) {
    }

    void init();

    // create for each element a scaled version of its geometry
    virtual void build_scaled_geometry(const Mesh::Geometry &mesh_geometry,
                                       std::map<Element, Mesh::Topology> &mesh_topologies,
                                       Mesh::Geometry &elem_geometry, scalar scale);

    virtual void build_scaled_topology(
        Mesh::Topology &mesh_topology,
        Mesh::Topology &triangles,
        Mesh::Topology &quads);

    virtual void get_scaled_wireframe(Mesh::Topology &mesh_topology, Mesh::Topology &lines);


    // convert element into quads and triangles
    virtual void convert_element(
        Mesh::Topology &mesh_topology,
        Mesh::Topology &triangles,
        Mesh::Topology &quads);

    virtual void convert_element_topo(int i_start,
                                      int num_elem,
                                      const Mesh::Topology &elem_topo,
                                      const Mesh::Topology &mesh_topologies,
                                      Mesh::Topology &topology);

    Mesh::Topology &topo_triangle() { return _topo_triangle; }
    Mesh::Topology &topo_quad() { return _topo_quad; }
    Mesh::Topology &topo_edge() { return _topo_edge; }
    Mesh::Geometry &geo_ref() { return _ref_element; }
    [[nodiscard]] FEM_Shape *shape() const { return _shape; }

    [[nodiscard]] virtual Element get_element_type() const = 0;

    virtual ~MeshConverter() { delete _shape; }

protected:
    [[nodiscard]] virtual FEM_Shape *get_shape() const = 0;

    // get the triangles in reference element
    [[nodiscard]] virtual Mesh::Topology get_elem_topo_triangle() const = 0;

    // get the quads in reference element
    [[nodiscard]] virtual Mesh::Topology get_elem_topo_quad() const = 0;

    // get the edges in reference element
    [[nodiscard]] virtual Mesh::Topology get_elem_topo_edges() const = 0;


    virtual void resize_topo(const int nb_elem, const int elem_topo_size, Mesh::Topology &topo) {
        topo.resize(topo.size() + nb_elem * elem_topo_size);
    }

    virtual void build_scaled_element_topo(int vid, int num_elem, const Mesh::Topology &elem_topo,
                                           Mesh::Topology &topology);

    FEM_Shape *_shape;
    Mesh::Topology _topo_triangle, _topo_quad, _topo_edge;
    Mesh::Geometry _ref_element;
};


struct TetraConverter : MeshConverter {
    [[nodiscard]] Element get_element_type() const override { return Tetra; }

    [[nodiscard]] Mesh::Topology get_elem_topo_edges() const override {
        return {0, 1, 1, 2, 2, 0, 0, 3, 1, 3, 2, 3};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_triangle() const override {
        return {0, 1, 3, 1, 2, 3, 0, 3, 2, 0, 2, 1};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_quad() const override { return {}; }
    [[nodiscard]] FEM_Shape *get_shape() const override { return new Tetra_4(); }
};

struct PyramidConverter : MeshConverter {
    [[nodiscard]] Element get_element_type() const override { return Pyramid; }


    [[nodiscard]] Mesh::Topology get_elem_topo_edges() const override {
        return {0, 1, 1, 2, 2, 3, 0, 4, 1, 4, 2, 4, 3, 4};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_triangle() const override {
        return {0, 1, 4, 3, 0, 4, 1, 2, 4, 2, 3, 4};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_quad() const override {
        return {3, 2, 1, 0};
    }

    [[nodiscard]] FEM_Shape *get_shape() const override { return new Pyramid_5(); }
};

struct PrysmConverter : MeshConverter {
    [[nodiscard]] Element get_element_type() const override { return Prism; }

    [[nodiscard]] Mesh::Topology get_elem_topo_triangle() const override {
        return {0, 1, 2, 3, 5, 4};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_edges() const override {
        return {0, 1, 1, 2, 2, 0, 0, 3, 1, 4, 2, 5, 3, 4, 4, 5, 5, 3};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_quad() const override {
        return {3, 4, 1, 0, 2, 5, 3, 0, 1, 4, 5, 2};
    }

    [[nodiscard]] FEM_Shape *get_shape() const override { return new Prism_6(); }
};

struct HexaConverter : MeshConverter {
    [[nodiscard]] Element get_element_type() const override { return Hexa; }


    [[nodiscard]] Mesh::Topology get_elem_topo_edges() const override {
        return {0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_quad() const override {
        return {0, 1, 2, 3,
                  4, 5, 1, 0,
                  5, 6, 2, 1,
                  3, 2, 6, 7,
                  7, 4, 0, 3,
                  7, 6, 5, 4};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_triangle() const override { return {}; }
    [[nodiscard]] FEM_Shape *get_shape() const override { return new Hexa_8(); }
};


struct Tetra10Converter : TetraConverter {
    [[nodiscard]] Element get_element_type() const override { return Tetra10; }

    [[nodiscard]] Mesh::Topology get_elem_topo_edges() const override {
        return {0, 4, 4, 1, 1, 5, 5, 2, 0, 6, 6, 2, 0, 7, 7, 3, 1, 8, 8, 3, 2, 9, 9, 3};
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_triangle() const override {
        return {
            0, 4, 7, 4, 1, 8, 4, 8, 7, 7, 8, 3,
            1, 5, 8, 5, 9, 8, 5, 2, 9, 8, 9, 3,
            2, 6, 9, 6, 0, 7, 6, 7, 9, 9, 7, 3,
            0, 6, 4, 6, 5, 4, 6, 2, 5, 4, 5, 1
        };
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_quad() const override {
        return {};
    }

    [[nodiscard]] FEM_Shape *get_shape() const override { return new Tetra_10(); }
};



struct Tetra20Converter : TetraConverter {
    [[nodiscard]] Element get_element_type() const override { return Tetra20; }

    [[nodiscard]] Mesh::Topology get_elem_topo_edges() const override {
        return
        {
            0, 4, 4, 5, 5, 1, 1, 6, 6, 7, 7, 2, 2, 8, 8, 9, 9, 0,
            0, 10, 10, 11, 11, 3, 3, 15, 15, 14, 14, 2, 1, 12, 12, 13, 13, 3
        };
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_triangle() const override {
        return {
            10, 0, 4, 10, 4, 16, 11, 10, 16, 11, 16, 13, 3, 11, 13, 16, 4, 5, 16, 5, 12, 13, 16, 12, 12, 5, 1,
            12, 1, 6, 12, 6, 17, 13, 12, 17, 13, 17, 15, 3, 13, 15, 17, 6, 7, 17, 7, 14, 15, 17, 14, 14, 7, 2,
            14, 2, 8, 14, 8, 18, 15, 14, 18, 15, 18, 11, 3, 15, 11, 18, 8, 9, 18, 9, 10, 11, 18, 10, 10, 9, 0,
            4, 0, 9, 4, 9, 19, 5, 4, 19, 5, 19, 6, 1, 5, 6, 19, 9, 8, 19, 8, 7, 6, 19, 7, 7, 8, 2

        };
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_quad() const override {
        return {};
    }


    [[nodiscard]] FEM_Shape *get_shape() const override { return new Tetra_20(); }
};

struct Hexa27Converter : HexaConverter {
    [[nodiscard]] Element get_element_type() const override { return Hexa27; }

    [[nodiscard]] Mesh::Topology get_elem_topo_edges() const override {
        return
        {
            0,8,8,1, 1,9,9,2, 2,10,10,3, 3,11,11,0, 0,12,12,4, 1,13,13,5, 2,14,14,6, 3,15,15,1, 4,16,16,5, 5,17,17,6, 6,18,18,7, 7,19,19,4
        };
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_triangle() const override {
        return { };
    }

    [[nodiscard]] Mesh::Topology get_elem_topo_quad() const override {
        return {
            3,10,20,11, 10,2,9,20, 11,20,8,0, 20,9,1,8,
            4,16,21,12, 16,5,13,21, 12,21,8,0, 21,13,1,8,
            5,17,22,13, 17,6,14,22, 13,22,9,1, 22,14,2,9,
            6,18,23,14, 18,7,15,23, 14,23,10,2, 23,15,3,10,
            7,19,24,15, 19,4,12,24, 15,24,11,3, 24,12,0,11,
            7,18,25,19, 18,6,17,25, 19,25,16,4, 25,17,5,16
        };
    }


    [[nodiscard]] FEM_Shape *get_shape() const override { return new Hexa_27(); }
};

inline MeshConverter* get_mesh_converter(const Element elem) {
    MeshConverter* converter = nullptr;
    switch (elem) {
        case Tetra: converter = new TetraConverter(); break;
        case Tetra10: converter = new Tetra10Converter();break;
        case Tetra20: converter = new Tetra20Converter();break;
        case Hexa: converter = new HexaConverter();break;
        case Pyramid: converter = new PyramidConverter();break;
        case Prism: converter = new PrysmConverter();break;
        case Hexa27: converter = new Hexa27Converter();break;
        default: converter = nullptr;
    }
    assert(converter != nullptr);
    if(converter != nullptr) converter->init();
    return converter;
}