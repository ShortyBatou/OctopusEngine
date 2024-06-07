#pragma once
#include <map>
#include <vector>
#include <algorithm>
#include "Core/Base.h"
#include "Mesh/mesh.h"

class MeshUtilities {
public:
    template<int nb>
    static void get_surface(
        std::set<Face<nb>>& faces,
        const Mesh::Topology& topology,
        const Mesh::Topology& ref_topology,
        const Mesh::Geometry& ref_geometry,
        bool remove_duplicates = true)
    {
        Mesh::Topology face_ids(nb);
        Mesh::Geometry face_vertices(nb);
        int nb_id_per_element = ref_topology.size();
        for (int i = 0; i < topology.size(); i += nb)
        {
            int element_id = i / nb_id_per_element;
            int face_id = (i % nb_id_per_element) / nb;
            for (int j = 0; j < nb; ++j) {
                face_ids[j] = topology[i + j];
                face_vertices[j] = ref_geometry[ref_topology[face_id * nb + j]];
            }
            try_add_face(Face<nb>(face_ids, face_vertices, element_id, face_id));

        }
    }

    template<int nb>
    void get_surface(
        std::set<Face<nb>>& faces,
        const Mesh::Topology& topology,
        const Mesh::Topology& ref_topology,
        bool remove_duplicates = true)
    {
        Mesh::Topology face_ids(nb);
        for (int i = 0; i < topology.size(); i += nb)
        {
            for (int j = 0; j < nb; ++j) face_ids[j] = topology[i + j];
            try_add_face(Face<nb>(face_ids, {}, i / nb_id_per_element));
        }
    }

private:
    template<int nb>
    static void try_add_face(std::set<Face<nb>>& faces, Face<nb>& face, bool remove_duplicates = true) {
        auto it = faces.find(face);
        if (it == faces.end())
            faces.insert(face);
        else if (remove_duplicates)
            faces.erase(it);
    }
};

