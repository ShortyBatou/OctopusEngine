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
        std::set<Face<NB>>& faces,
        const Mesh::Topology& topology,
        const Mesh::Topology& ref_topology,
        bool remove_duplicates = true)
    {
        int nb_id_per_element = ref_topology.size();
        Mesh::Topology face_ids(NB);
        for (int i = 0; i < topology.size(); i += NB)
        {
            int element_id = i / nb_id_per_element;
            int face_id = (i % nb_id_per_element) / NB;
            for (int j = 0; j < NB; ++j) face_ids[j] = topology[i + j];
            MeshTools::TryAddFace(faces, Face<NB>(face_ids, {}, element_id, face_id), remove_duplicates);
        }
    }

    // get face that correspond to surface and add face geometry info
    template<int NB>
    static void GetSurface(
        std::set<Face<NB>>& faces,
        const Mesh::Topology& topology,
        const Mesh::Topology& ref_topology,
        const Mesh::Geometry& ref_geometry,
        bool remove_duplicates = true)
    {
        Mesh::Topology face_ids(NB);
        Mesh::Geometry face_vertices(NB);
        int nb_id_per_element = ref_topology.size();
        for (int i = 0; i < topology.size(); i += NB)
        {
            int element_id = i / nb_id_per_element;
            int face_id = (i % nb_id_per_element) / NB;
            for (int j = 0; j < NB; ++j) {
                face_ids[j] = topology[i + j];
                face_vertices[j] = ref_geometry[ref_topology[face_id * NB + j]];
            }
            MeshTools::TryAddFace(faces, Face<NB>(face_ids, face_vertices, element_id, face_id), remove_duplicates);
        }
    }

    /// removes duplicate between two face sets
    template<int NB>
    static void RemoveDuplicates(std::set<Face<NB>>& a_faces, std::set<Face<NB>>& b_faces) {
        for (const Face<NB>& face : a_faces) {
            auto it = b_faces.find(face);
            if (it != b_faces.end()) {
                a_faces.erase(face);
                b_faces.erase(it);
            }
        }
    }

    /// convert the face set into an indices array
    template<int NB>
    static void ExtractTopo(const std::set<Face<NB>>& faces, Mesh::Topology& topology) {
        topology.resize(NB * faces.size());
        Mesh::Topology ids(NB);
        int i = 0;
        for (const Face<NB>& face : faces) {
            for (int j = 0; j < NB; ++j) {
                topology[i * NB + j] = face.ids[j];
            }
            i++;
        }
    }

    /// gives the map between a face and its element id
    template<int NB>
    static void ExtractFaceToElem(const std::set<Face<NB>>& faces, Mesh::Topology& face_to_elem) {
        int f_size = face_to_elem.size();
        face_to_elem.resize(f_size + faces.size());
        int i = 0;
        for (const Face<NB>& face : faces) {
            face_to_elem[f_size + i] = face.element_id;
            i++;
        }
    }

    // get the map face -> element from elements topo
    template<int NB>
    static void ExtractTopoToElem(const Mesh::Topology& topo, const Mesh::Topology& ref_topology, Mesh::Topology& face_to_elem) {
        int nb_id_per_element = ref_topology.size();
        face_to_elem.resize(topo.size() / NB);
        for (int i = 0; i < topo.size(); i += NB)
            face_to_elem[i / NB] = i / nb_id_per_element;
    }

    // get the geometry according to faces, give a map vertices -> element and old to new ids
    template<int NB>
    static void RebuildFaces(std::set<Face<NB>>& faces, Mesh::Geometry& geometry, std::vector<int>& v_to_elem, std::map<int, int>& map_id) {
        std::set<Face<NB>> new_faces;
        Mesh::Topology ids(NB);
        for (const Face<NB>& face : faces) {
            
            for (int i = 0; i < NB; ++i) {
                int id = face.ids[i];
                auto it = map_id.find(id);
                // if not found
                if (it == map_id.end()) {
                    map_id[id] = geometry.size();
                    id = geometry.size();
                    geometry.push_back(face.vertices[i]);
                    v_to_elem.push_back(face.element_id);
                }
                else {
                    id = map_id[id];
                }

                ids[i] = id;
            }
            new_faces.insert(Face<NB>(ids, face.vertices, face.element_id, face.face_id));
        }    
        faces = new_faces;
    }

    // subdibise faces according to a pattern
    template<int NB>
    static void Subdivise(
        const Mesh::Topology& subdivision_pattern,
        const Mesh::Topology& subdivision_edge,
        std::set<Face<NB>>& faces,
        std::set<Face<2>>& wireframe,
        std::map<Face<2>, int>& edges,
        Mesh::Geometry& refined_geometry,
        std::vector<int>& v_element_id
    ) {
        int nb_triangles = faces.size();

        Mesh::Topology f_ids(NB + subdivision_edge.size()/2);
        Mesh::Geometry f_verts(f_ids.size());

        Mesh::Topology new_f_ids(NB);
        Mesh::Geometry new_f_verts(new_f_ids.size());
        std::set<Face<NB>> new_faces;
        // subdivise each triangle in mesh
        for (const Face<NB>& face : faces)
        {
            int i = 0;
            for (; i < NB; ++i) {
                f_ids[i] = face.ids[i]; // copy the face ids
                f_verts[i] = face.vertices[i];
            }

            // Apply the edge subdivision pattern
            for (int j = 0; j < subdivision_edge.size(); j += 2) {
                int a = subdivision_edge[j];
                int b = subdivision_edge[j + 1];
                Vector3 center = (f_verts[a] + f_verts[b]) * 0.5f;

                // find if this edge has already been subdivided
                // no => add the new vertice to geometry and add edge
                // yes => get the vertice
                Face<2> edge({ f_ids[a], f_ids[b] });
                int cid;
                if (edges.find(edge) == edges.end()) {
                    cid = refined_geometry.size();
                    refined_geometry.push_back(center);
                    v_element_id.push_back(face.element_id);
                    edges[edge] = cid;

                    if (wireframe.find(edge) != wireframe.end()) {
                        wireframe.erase(edge);
                        wireframe.insert(Face<2>({ f_ids[a], cid }));
                        wireframe.insert(Face<2>({ f_ids[b], cid }));
                    }
                }
                else {
                    cid = edges[edge];
                }
                f_ids[i] = cid;
                f_verts[i] = center;
                i++;
            }

            // Apply the face subdivision pattern to create new faces
            for (int j = 0; j < subdivision_pattern.size(); j += NB) {

                for (int k = 0; k < NB; ++k) {
                    new_f_ids[k] = f_ids[subdivision_pattern[j + k]];
                    new_f_verts[k] = f_verts[subdivision_pattern[j + k]];
                }
                new_faces.insert(Face<NB>(new_f_ids, new_f_verts, face.element_id, face.face_id));
            }
        }
        faces = new_faces;
    }

private:
    template<int nb>
    static void TryAddFace(std::set<Face<nb>>& faces, Face<nb>& face, bool remove_duplicates) {
        auto it = faces.find(face);
        if (it == faces.end())
            faces.insert(face);
        else if (remove_duplicates)
            faces.erase(it);
    }
};

