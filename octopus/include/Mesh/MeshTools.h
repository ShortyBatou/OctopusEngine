#pragma once
#include <map>
#include <vector>
#include <algorithm>
#include "Core/Base.h"
#include "Mesh/mesh.h"
#include <set>

class MeshTools {
public:
    // get face that correspond to surface
    template<int NB>
    static void GetSurface(
        std::set<Face<NB> > &faces,
        const Mesh::Topology &topology,
        const Mesh::Topology &ref_topology,
        bool remove_duplicates = true);

    // get face that correspond to surface and add face geometry info
    template<int NB>
    static void GetSurface(
        std::set<Face<NB> > &faces,
        const Mesh::Topology &topology,
        const Mesh::Topology &ref_topology,
        const Mesh::Geometry &ref_geometry,
        bool remove_duplicates = true);

    /// removes duplicate between two face sets
    template<int NB>
    static void RemoveDuplicates(std::set<Face<NB> > &a_faces, std::set<Face<NB> > &b_faces);

    /// convert the face set into an indices array
    template<int NB>
    static void ExtractTopo(const std::set<Face<NB> > &faces, Mesh::Topology &topology);

    /// gives the map between a face and its element id
    template<int NB>
    static void ExtractFaceToElem(const std::set<Face<NB> > &faces, Mesh::Topology &face_to_elem);

    // get the map face -> element from elements topo
    template<int NB>
    static void ExtractTopoToElem(const Mesh::Topology &topo, const Mesh::Topology &ref_topology,
                                  Mesh::Topology &face_to_elem);

    // get the geometry according to faces, give a map vertices -> element and old to new ids
    template<int NB>
    static void RebuildFaces(std::set<Face<NB> > &faces, Mesh::Geometry &geometry, std::vector<int> &v_to_elem,
                             std::map<int, int> &map_id);

    // subdibise faces according to a pattern
    template<int NB>
    static void Subdivise(
        const Mesh::Topology &subdivision_pattern,
        const Mesh::Topology &subdivision_edge,
        std::set<Face<NB> > &faces,
        std::set<Face<2> > &wireframe,
        std::map<Face<2>, int> &edges,
        Mesh::Geometry &refined_geometry,
        std::vector<int> &v_element_id
    );

private:
    template<int nb>
    static void TryAddFace(std::set<Face<nb> > &faces, Face<nb> &face, bool remove_duplicates);
};
