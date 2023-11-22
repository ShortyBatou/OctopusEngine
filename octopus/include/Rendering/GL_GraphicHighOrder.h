#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"
#include <map>
#include <set>

template<unsigned int NB>
struct Ref_Element {
    Vector3 vertices[NB];
    unsigned int element[NB];
};

class GL_GraphicHighOrder : public GL_Graphic
{
    using Edge = std::pair<unsigned int, unsigned int>;
public:
    GL_GraphicHighOrder(unsigned int quality, const Color& color = Color(0.8, 0.4, 0.4, 1.0)) : GL_Graphic(color), _quality(quality)
    {
        _converters[Tetra10] = new Tetra10Converter();
        for (auto& elem : _converters) elem.second->init();
    }
    virtual void update() override
    {
        if (_mesh->need_update() || _mesh->has_dynamic_topology())
        {
            update_buffer_topology();
        }

        if (_mesh->need_update() || _mesh->has_dynamic_geometry())
        {
            update_buffer_geometry();
            if (_multi_color) update_buffer_colors();
        }

        _mesh->need_update() = false;
    }


    virtual void update_buffer_geometry() override {
        auto& tetra10 = _mesh->topology(Tetra10);
        FEM_Shape* shape = _converters[Tetra10]->shape();
        Mesh::Geometry geometry(refined_geometry.size());
        for (unsigned int i = 0; i < refined_geometry.size(); ++i) {
            std::vector<scalar> N_v = shape->build_shape(refined_geometry[i].x, refined_geometry[i].y, refined_geometry[i].z);
            geometry[i] = Unit3D::Zero();
            for (unsigned int j = 0; j < N_v.size(); ++j) {
                geometry[i] += N_v[j] * _mesh->geometry()[tetra10[v_tetra_id[i] * 10 + j]];
            }
        }
        this->_b_vertex->load_data(geometry);
    }

    virtual void update_buffer_topology() override
    {
        std::map<Element, Mesh::Topology> elem_topologies;
        // convert tetra10 into quads and triangles
        _converters[Tetra10]->convert_element(_mesh->topologies(), elem_topologies);
        
        // get surface triangles associated with their tetra and face num and their coordinate in referential element
        std::vector<Face<3>> triangles = get_surface<3>(
            elem_topologies[Triangle], 
            _converters[Tetra10]->geo_ref(), 
            _converters[Tetra10]->topo_triangle()
        );


        // get surface vertices and their tetra_id
        refined_geometry.clear();
        v_tetra_id.clear();

        std::map<unsigned int, unsigned int> map_id; // temp
        for (Face<3>&tri : triangles) {
            for (unsigned int i = 0; i < tri.ids.size(); ++i) {
                unsigned int id = tri.ids[i];
                auto it = map_id.find(id);
                // if not found
                if (it == map_id.end()) {
                    map_id[id] = refined_geometry.size();
                    id = refined_geometry.size();
                    refined_geometry.push_back(tri.vertices[i]);
                    v_tetra_id.push_back(tri.element_id);
                }
                else {
                    id = map_id[id];
                }

                tri.ids[i] = id;
            }
            tri = Face<3>(tri.ids, tri.vertices, tri.element_id, tri.face_id);
        }

        using Edge = std::pair<unsigned int, unsigned int>;
        // triangle subdivision pattern;
        std::vector<unsigned int> subdivision_pattern = { 0,3,5, 3,1,4, 3,4,5, 5,4,2 }; 
        std::vector<Edge> subdivision_edges = { Edge(0,1), Edge(1,2), Edge(0,2) }; 

        // TODO quad subdivision patter
        
        for (unsigned int i = 0; i < _quality; ++i) {
            std::map<Edge, unsigned int> edges;
            subdivise<3>(subdivision_pattern, subdivision_edges, triangles, edges, v_tetra_id, refined_geometry);
            // TODO quad
        }

        std::vector<unsigned int> tri(triangles.size()*3);
        for (unsigned int i = 0; i < triangles.size(); ++i) {
            tri[i * 3] = triangles[i].ids[0];
            tri[i * 3+1] = triangles[i].ids[1];
            tri[i * 3+2] = triangles[i].ids[2];
        }

        
        

        if (tri.size() > 0)
            this->_b_triangle->load_data(tri);
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

                Edge edge(faces[i].ids[sub_edge.first], faces[i].ids[sub_edge.second]);

                Vector3 pa = v_tri[sub_edge.first];
                Vector3 pb = v_tri[sub_edge.second];
                Vector3 c = (pa + pb) * scalar(0.5);
                // if the edge allready exist, get the middle point id, else, create the middle point
                unsigned int cid;
                auto it = edges.find(edge);
                if (it == edges.end()) {
                    cid = refined_geometry.size();
                    refined_geometry.push_back(c);
                    v_element_id.push_back(faces[i].element_id);
                    edges[edge] = cid;
                }
                else {
                    cid = edges[edge];
                }

                v_ids[j] = cid;
                v_tri[j] = c;

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