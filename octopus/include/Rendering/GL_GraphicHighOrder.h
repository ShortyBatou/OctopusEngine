#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"
#include <map>
#include <set>


class GL_GraphicHighOrder : public GL_Graphic
{
    using Edge = std::pair<unsigned int, unsigned int>;
public:
    GL_GraphicHighOrder(unsigned int quality, const Color& color = Color(0.8, 0.4, 0.4, 1.0)) : GL_Graphic(color), _quality(quality)
    {
        _converters[Tetra] = new TetraConverter();
        _converters[Tetra10] = new Tetra10Converter();
        for (auto& elem : _converters) elem.second->init();
    }


    virtual void get_geometry(Mesh::Geometry& geometry) override {

        Element elem;
        for (auto& t : _mesh->topologies()) {
            if (t.second.size() > 0) {
                elem = t.first;
                break;
            }
        }
        unsigned int elem_size = element_vertices(elem);

        auto& tetras = _mesh->topology(elem);
        FEM_Shape* shape = _converters[elem]->shape();
        geometry.resize(refined_geometry.size());
        for (unsigned int i = 0; i < refined_geometry.size(); ++i) {
            std::vector<scalar> N_v = shape->build_shape(refined_geometry[i].x, refined_geometry[i].y, refined_geometry[i].z);
            geometry[i] = Unit3D::Zero();
            for (unsigned int j = 0; j < N_v.size(); ++j) {
                geometry[i] += N_v[j] * _mesh->geometry()[tetras[v_tetra_id[i] * elem_size + j]];
            }
        }
    }

    virtual void get_topology(Mesh::Topology& lines, Mesh::Topology& triangles, Mesh::Topology& quads) override
    {

        // triangle subdivision pattern;
        std::vector<unsigned int> subdivision_pattern = { 0,3,5, 3,1,4, 3,4,5, 5,4,2 };
        std::vector<Edge> subdivision_edges = { Edge(0,1), Edge(1,2), Edge(0,2) };
        // convert tetra10 into quads and triangles

        Element elem;
        for (auto& t : _mesh->topologies()) {
            if (t.second.size() > 0) {
                elem = t.first;
                break;
            }
        }
        unsigned int elem_size = element_vertices(elem);
        Mesh::Topology elem_triangles, elem_quads;
        _converters[elem]->convert_element(_mesh->topologies(), elem_triangles, elem_quads);

        // get surface triangles associated with their tetra and face num and their coordinate in referential element
        std::vector<Face<3>> surface_triangles = get_surface<3>(
            elem_triangles,
            _converters[elem]->geo_ref(),
            _converters[elem]->topo_triangle()
        );

        auto& elements = _mesh->topology(elem);
        auto& elem_edges = _converters[elem]->topo_edge();
        
        std::set<Edge> wireframe;
        build_wireframe(elem_size, elements, elem_edges, subdivision_edges, surface_triangles, wireframe);

        // change surface face ids to only use surface vertices and get their tetra_id
        std::map<unsigned int, unsigned int> map_id;
        rebuild_faces<3>(surface_triangles, map_id);

        // change vertices ids for wireframe
        rebuild_wireframe(map_id, wireframe);
        

        // TODO quad subdivision pattern
        for (unsigned int i = 0; i < _quality; ++i) {
            std::map<Edge, unsigned int> edges;
            subdivise<3>(subdivision_pattern, subdivision_edges, surface_triangles, edges, wireframe, v_tetra_id, refined_geometry);
            // TODO quad
        }

        // we use quads list to store triangle because we don't want wireframe ! (automatic for triangles)
        quads.resize(surface_triangles.size()*3);
        for (unsigned int i = 0; i < surface_triangles.size(); ++i) {
            quads[i * 3] = surface_triangles[i].ids[0];
            quads[i * 3+1] = surface_triangles[i].ids[1];
            quads[i * 3+2] = surface_triangles[i].ids[2];
        } 

        lines.resize(wireframe.size() * 2);
        unsigned int e = 0;
        for (const Edge& edge : wireframe) {
            lines[e] = edge.first;
            lines[e + 1] = edge.second;
            e+=2;
        }
    }

    // nb = face nb vertices
    // ref_geometry = refence element geometry 
    // ref_topology = refence element topology 
    // faces        = surface faces
    template<unsigned int nb> 
    std::vector<Face<nb>> get_surface(
        const Mesh::Topology& topology,
        const Mesh::Geometry& ref_geometry,
        const Mesh::Topology& ref_topology)
    {
        std::set<Face<nb>> faces;
        Mesh::Topology face_ids(nb);
        Mesh::Geometry face_vertices(nb);
        unsigned int nb_id_per_element = ref_topology.size();
        for (unsigned int i = 0; i < topology.size(); i += nb)
        { 
            unsigned int element_id = i / nb_id_per_element;
            unsigned int face_id = (i % nb_id_per_element) / nb;

            // get the face in element
            // get the face vertices in referece geometry 
            for (unsigned int j = 0; j < nb; ++j) 
            {
                face_ids[j] = topology[i + j];
                face_vertices[j] = ref_geometry[ref_topology[face_id * nb + j]];
            }

            // check if face allready exist, if not add it to faces, if yes, delete both faces
            Face<nb> face(face_ids, face_vertices, element_id, face_id);
            auto it = faces.find(face);
            if (it == faces.end())
                faces.insert(face);
            else
                faces.erase(it);
        }

        return std::vector<Face<nb>>(faces.begin(), faces.end());
    }

    void build_wireframe(unsigned int element_size,
        const Mesh::Topology& elements,
        const Mesh::Topology& elem_edges,
        const std::vector<Edge>& subdivision_edges,
        const std::vector<Face<3>>& triangles,
        std::set<Edge>& wireframe) {
        // for each triangle, get the element id, if found allready don't add
        std::set<unsigned int> surface_element;
        std::set<Edge> wireframe_elem;
        for (const Face<3>&tri : triangles) {
            if (surface_element.find(tri.element_id) == surface_element.end()) {
                surface_element.insert(tri.element_id);
                for (unsigned int i = 0; i < elem_edges.size(); i += 2) {
                    unsigned int a = elements[tri.element_id * element_size + elem_edges[i]];
                    unsigned int b = elements[tri.element_id * element_size + elem_edges[i + 1]];
                    Edge edge = (a < b) ? Edge(a, b) : Edge(b, a);
                    wireframe_elem.insert(edge);
                }
            }
        }

        for (const Face<3>&tri : triangles) {
            for (const Edge& e : subdivision_edges) {
                unsigned int a = tri.ids[e.first];
                unsigned int b = tri.ids[e.second];
                Edge edge = (a < b) ? Edge(a, b) : Edge(b, a);
                auto it = wireframe_elem.find(edge);
                if (it == wireframe_elem.end()) continue;
                wireframe.insert(edge);
            }
        }
    }

    template<unsigned int NB>
    void rebuild_faces(std::vector<Face<NB>>& faces, std::map<unsigned int, unsigned int>& map_id) {
        refined_geometry.clear();
        v_tetra_id.clear();
        for (Face<NB>&face : faces) {
            for (unsigned int i = 0; i < NB; ++i) {
                unsigned int id = face.ids[i];
                auto it = map_id.find(id);
                // if not found
                if (it == map_id.end()) {
                    map_id[id] = refined_geometry.size();
                    id = refined_geometry.size();
                    refined_geometry.push_back(face.vertices[i]);
                    v_tetra_id.push_back(face.element_id);
                }
                else {
                    id = map_id[id];
                }

                face.ids[i] = id;
            }
            face = Face<NB>(face.ids, face.vertices, face.element_id, face.face_id);
        }
    }

    void rebuild_wireframe(std::map<unsigned int, unsigned int>& map_id, std::set<Edge>& wireframe) {
        std::set<Edge> remaped_wireframe;
        for (const Edge& edge : wireframe) {
            unsigned int a = map_id[edge.first];
            unsigned int b = map_id[edge.second];
            Edge edge = (a < b) ? Edge(a, b) : Edge(b, a);
            remaped_wireframe.insert(edge);
        }
        wireframe = remaped_wireframe;
    }

    // subdivision_pattern = which triangles are created when all edges have been splited
    // subdivision_edges   = which edges must be splited
    // faces               = surface faces of mesh
    // edges               = edges that are allready existing in this subdivision
    // v_element_id        = vertice's element id to know which element influence the vertice
    // refined_geometry    = surface mesh geometry
    template<unsigned int NB>
    void subdivise(
        const Mesh::Topology& subdivision_pattern,
        const std::vector<Edge>& subdivision_edges,
        std::vector<Face<NB>>& faces,
        std::map<Edge, unsigned int>& edges,
        std::set<Edge>& wireframe,
        std::vector<unsigned int>& v_element_id,
        Mesh::Geometry& refined_geometry
    ) {
        //// SUBDIVISION FUNCTION
        
         // existing edges and it's associated mid point
        unsigned int nb_triangles = faces.size();
        Mesh::Topology face_topo(NB);
        Mesh::Geometry face_vertices(NB);
        // subdivise each triangle in mesh
        for (unsigned int i = 0; i < nb_triangles; ++i)
        {
            unsigned int j = 0;
            std::vector<unsigned int> v_ids(NB + subdivision_edges.size()); // nb vertices in subdivision
            std::vector<Vector3> v_tri(v_ids.size());
            for (; j < faces[i].ids.size(); ++j) {
                v_ids[j] = faces[i].ids[j]; // copy the face ids
                v_tri[j] = faces[i].vertices[j];
            }

            // for each edge, find or create the middle point
            for (const Edge& sub_edge : subdivision_edges) {

                unsigned int a = faces[i].ids[sub_edge.first];
                unsigned int b = faces[i].ids[sub_edge.second];
                
                Edge edge = (a < b) ? Edge(a, b) : Edge(b,a);

                Vector3 pa = v_tri[sub_edge.first];
                Vector3 pb = v_tri[sub_edge.second];
                Vector3 center = (pa + pb) * scalar(0.5);
                // if the edge allready exist, get the middle point id, else, create the middle point
                unsigned int cid;
                auto it = edges.find(edge);
                if (it == edges.end()) {
                    cid = refined_geometry.size();
                    refined_geometry.push_back(center);
                    v_element_id.push_back(faces[i].element_id);
                    edges[edge] = cid;

                    if (wireframe.find(edge) != wireframe.end()) {
                        wireframe.erase(edge);
                        Edge w1 = (a < cid) ? Edge(a, cid) : Edge(cid, a);
                        Edge w2 = (b < cid) ? Edge(b, cid) : Edge(cid, b);
                        wireframe.insert(w1);
                        wireframe.insert(w2);
                    }
                   
                }
                else {
                    cid = edges[edge];
                }

                v_ids[j] = cid;
                v_tri[j] = center;

                j++;
            }

            for (unsigned int k = 0; k < subdivision_pattern.size(); k += NB) {

                for (unsigned int l = 0; l < NB; ++l) {
                    face_topo[l] = v_ids[subdivision_pattern[k + l]];
                    face_vertices[l] = v_tri[subdivision_pattern[k + l]];
                }

                Face<NB> new_tri(face_topo, face_vertices, faces[i].element_id, faces[i].face_id);

                if (k == 0) {
                    faces[i] = new_tri;
                }
                else {
                    faces.push_back(new_tri);
                }
            }
        }
    }

protected:
    Mesh::Geometry refined_geometry;
    std::vector<unsigned int> v_tetra_id;
    unsigned int _quality;
    std::map<Element, MeshConverter*> _converters; 
};