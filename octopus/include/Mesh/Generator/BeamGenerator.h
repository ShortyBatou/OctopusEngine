#pragma once
#include "Mesh/Generator/MeshGenerator.h"
#include "Mesh/Converter/MeshConverter.h"
struct BeamMeshGenerator : public MeshGenerator
{
    BeamMeshGenerator(const Vector3I& subdivisions, const Vector3& sizes)
        : _subdivisions(subdivisions + Vector3I(1)), _sizes(sizes)
    { 
        _x_step = _sizes.x / scalar(_subdivisions.x - 1);
        _y_step = _sizes.y / scalar(_subdivisions.y - 1);
        _z_step = _sizes.z / scalar(_subdivisions.z - 1);
    }
    

    virtual Mesh* build() override
    {
        Mesh* mesh = new Mesh();
        buildGeometryGrid(mesh->geometry());
        apply_transform(mesh->geometry());
        int ids[8];
        for (int z = 0; z < _subdivisions.z - 1; ++z)
        for (int y = 0; y < _subdivisions.y - 1; ++y)
        for (int x = 0; x < _subdivisions.x - 1; ++x)
        {
            this->get_cell_vertices_ids(x, y, z, ids);
            addGeometryAtCell(x, y, z, mesh->geometry());
            buildTopoAtCell(ids, mesh->topologies());
        }
        return mesh;
    }

    void buildGeometryGrid(Mesh::Geometry& geometry)
    {
        for (int z = 0; z < _subdivisions.z; ++z)
        for (int y = 0; y < _subdivisions.y; ++y)
        for (int x = 0; x < _subdivisions.x; ++x)
            geometry.push_back(Vector3(x * _x_step, y * _y_step, z * _z_step));
    }

    void get_cell_vertices_ids(int x, int y, int z, int* ids)
    {
        ids[0] = icoord_to_id(x, y, z);
        ids[1] = icoord_to_id(x + 1, y, z);
        ids[2] = icoord_to_id(x + 1, y, z + 1);
        ids[3] = icoord_to_id(x, y , z+ 1);

        ids[4] = icoord_to_id(x, y + 1, z);
        ids[5] = icoord_to_id(x + 1, y + 1, z);
        ids[6] = icoord_to_id(x + 1, y + 1, z + 1);
        ids[7] = icoord_to_id(x, y + 1, z + 1);
    }

    int icoord_to_id(int x, int y, int z)
    {
        return x + y * _subdivisions.x + z * _subdivisions.y * _subdivisions.x;
    }

    virtual void addGeometryAtCell(int x, int y, int z, Mesh::Geometry& geometry) { }
    virtual void buildTopoAtCell(int ids[8], std::map<Element, Mesh::Topology>& topologies) = 0;

protected:
    Vector3I _subdivisions;
    Vector3 _sizes;
    scalar _x_step, _y_step, _z_step;
};

class HexaBeamGenerator : public BeamMeshGenerator
{
public:

    HexaBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes) 
        : BeamMeshGenerator(_subdivisions, _sizes)
    { }

    virtual void buildTopoAtCell(int ids[8], std::map<Element, Mesh::Topology>& topologies) override
    {
        for (int i = 0; i < 8; ++i) 
            topologies[Hexa].push_back(ids[i]);
    }

    virtual ~HexaBeamGenerator() { }
};

class PrismBeamGenerator : public BeamMeshGenerator
{
public:

    PrismBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes)
        : BeamMeshGenerator(_subdivisions, _sizes)
    { }

    virtual void buildTopoAtCell(int ids[8], std::map<Element, Mesh::Topology>& topologies)
    {
        static int prysms[12] {0, 1, 3, 4, 5, 7, 
                                    1, 2, 3, 5, 6, 7 
        };
        for (int i = 0; i < 12; ++i)
            topologies[Prism].push_back(ids[prysms[i]]);
    }

    virtual ~PrismBeamGenerator() { }
};

class PyramidBeamGenerator : public BeamMeshGenerator
{
public:
    int _mid_id;

    PyramidBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes)
        : BeamMeshGenerator(_subdivisions, _sizes), _mid_id(0)
    { }

    void addGeometryAtCell(int x, int y, int z, Mesh::Geometry& geometry) override
    {
        _mid_id = geometry.size();
        Vector3 v(scalar(x * this->_x_step + this->_x_step * 0.5),
                  scalar(y * this->_y_step + this->_y_step * 0.5),
                  scalar(z * this->_z_step + this->_z_step * 0.5));
        geometry.push_back(v);
    }

    virtual void buildTopoAtCell(int ids[8], std::map<Element, Mesh::Topology>& topologies) override
    {
        static int pyramids[30] {3, 2, 1, 0, 8,
                                      0, 1, 5, 4, 8, 
                                      1, 2, 6, 5, 8, 
                                      2, 3, 7, 6, 8,
                                      3, 0, 4, 7, 8,
                                      4, 5, 6, 7, 8};
        for (int i = 0; i < 30; ++i)
        {
            int id;
            if (pyramids[i] == 8) id = _mid_id;
            else id = ids[pyramids[i]];
            topologies[Pyramid].push_back(id);
        }
    }
    virtual ~PyramidBeamGenerator() { }
};


class TetraBeamGenerator : public BeamMeshGenerator
{
public:

    TetraBeamGenerator(const Vector3I& _subdivisions, const Vector3& _sizes)
        : BeamMeshGenerator(_subdivisions, _sizes)
    { } 

    virtual void buildTopoAtCell(int ids[8], std::map<Element, Mesh::Topology>& topologies) override
    {
        //static unsigned tetras[24]{ 0,4,6,5, 0,4,7,6, 0,7,3,6, 2,0,3,6, 2,0,6,1, 6,0,5,1 }; int nb = 24;
        static int tetras[24]{ 0,4,6,5, 3,6,2,0, 0,4,7,6, 3,6,0,7, 2,0,6,1, 6,0,5,1 }; int nb = 24;
        //static unsigned tetras[20]{ 1,6,5,4, 1,2,6,3, 0,1,4,3, 7,6,3,4, 1,3,6,4 }; int nb = 20;
        //static unsigned tetras[20]{ 0,3,2,7, 7,4,5,0, 7,6,2,5, 2,1,0,5, 7,0,5,0}; int nb = 20;
        for (int i = 0; i < nb; ++i)
            topologies[Tetra].push_back(ids[tetras[i]]);
    }
    virtual ~TetraBeamGenerator() { }
};


void subdive_tetra(Mesh::Geometry& geometry, std::map<Element, Mesh::Topology>& topologies) {
    Mesh::Topology tetras = topologies[Tetra];
    topologies[Tetra].clear();
    topologies[Tetra10].clear();

    using Edge = std::pair<int, int>;
    int tetra_10_topo[32] = { 0,4,6,7, 1,5,4,8, 7,8,9,3, 2,6,5,9, 6,4,5,7, 7,4,5,8, 6,5,9,7, 7,8,5,9 };
    Edge tetra_edges[6] = { Edge(0,1), Edge(1,2), Edge(0,2), Edge(0,3), Edge(1,3), Edge(2,3) };
    int e1, e2;
    Edge e;
    std::map<Edge, unsigned int> edges;
    for (int i = 0; i < tetras.size(); i += 4) {
        int ids[10];
        int j = 0;
        for (; j < 4; ++j) ids[j] = tetras[i + j];

        for (Edge& tet_e : tetra_edges) {
            e1 = ids[tet_e.first]; e2 = ids[tet_e.second];
            if (e1 > e2) std::swap(e1, e2);
            e.first = e1; e.second = e2;

            int id;
            // edge found in map
            if (edges.find(e) != edges.end()) {
                id = edges[e];
            }
            else {
                id = geometry.size();
                Vector3 pa = Vector3(geometry[e1]);
                Vector3 pb = Vector3(geometry[e2]);

                Vector3 p = scalar(0.5) * (pa + pb);
                geometry.push_back(p);
                edges[e] = id;
            }
            ids[j] = id;
            ++j;
        }

        for (int k = 0; k < 32; ++k) {
            topologies[Tetra].push_back(ids[tetra_10_topo[k]]);
        }

    }
}

void tetra4_to_tetra10(Mesh::Geometry& geometry, std::map<Element, Mesh::Topology>& topologies)
{
    Mesh::Topology& tetras = topologies[Tetra];
    
    topologies[Tetra10].clear();

    using Edge = std::pair<int, int>;
    Edge tetra_edges[6] = { Edge(0,1), Edge(1,2), Edge(0,2), Edge(0,3), Edge(1,3), Edge(2,3) };
    int e1, e2;
    Edge e;
    std::map<Edge, int> edges;
    for (int i = 0; i < tetras.size(); i += 4) {
        int ids[10];
        int j = 0;
        for (; j < 4; ++j) ids[j] = tetras[i + j];

        for (Edge& tet_e : tetra_edges) {
            e1 = ids[tet_e.first]; e2 = ids[tet_e.second];
            if (e1 > e2) std::swap(e1, e2);
            e.first = e1; e.second = e2;

            int id;
            // edge found in map
            if (edges.find(e) != edges.end()) {
                id = edges[e];
            }
            else {
                id = geometry.size();
                Vector3 pa = Vector3(geometry[e1]);
                Vector3 pb = Vector3(geometry[e2]);

                Vector3 p = scalar(0.5) * (pa + pb);
                geometry.push_back(p);
                edges[e] = id;
            }
            ids[j] = id;
            ++j;
        }

        for (int k = 0; k < 10; ++k) {
            topologies[Tetra10].push_back(ids[k]);
        }
    }
    topologies[Tetra].clear();
}

void tetra4_to_tetra20(Mesh::Geometry& geometry, std::map<Element, Mesh::Topology>& topologies)
{
    Mesh::Topology& tetras = topologies[Tetra];
    topologies[Tetra20].clear();
    int nb = elem_nb_vertices(Tetra20);

    std::vector<int> ids(20);

    TetraConverter tetra_converter;
    const Mesh::Topology& tetra_edges = tetra_converter.get_elem_topo_edges();
    const Mesh::Topology& tetra_faces = tetra_converter.get_elem_topo_triangle();

    std::map<Face<2>, std::vector<int>> existing_edges;
    std::map<Face<3>, std::vector<int>> existing_faces;

    std::vector<int> v_edge_ids(2);
    std::vector<int> v_face_ids(1);

    for (int i = 0; i < tetras.size(); i += 4) {
        int j = 0;
        for (; j < 4; ++j) ids[j] = tetras[i + j];

        // edges
        for (int k = 0; k < tetra_edges.size(); k+=2) {
            int e_a = ids[tetra_edges[k]];
            int e_b = ids[tetra_edges[k+1]];
            Face<2> edge({ e_a, e_b });
            
            auto it = existing_edges.find(edge);
            // edge found in map
            if (it != existing_edges.end()) {
                v_edge_ids = existing_edges[edge];
               
                if (edge.ids[0] != it->first.ids[0])
                    std::reverse(v_edge_ids.begin(), v_edge_ids.end());
            }
            else {
                for (int w = 0; w < 2; w++) {
                    scalar weight = scalar(w+1) / scalar(3);
                    Vector3 p = geometry[e_a] * (scalar(1.) - weight) + geometry[e_b] * weight;
                    v_edge_ids[w] = geometry.size();
                    geometry.push_back(p);
                }
                existing_edges[edge] = v_edge_ids;
            }

            for (int e_id : v_edge_ids) {
                ids[j] = e_id;
                ++j;
            }
        }

        //faces
        for (int k = 0; k < tetra_faces.size(); k += 3) {
            int f_a = ids[tetra_faces[k]];
            int f_b = ids[tetra_faces[k + 1]];
            int f_c = ids[tetra_faces[k + 2]];
            Face<3> face({ f_a, f_b, f_c });
            // edge found in map
            if (existing_faces.find(face) != existing_faces.end()) {
                v_face_ids = existing_faces[face];
                // only works for P3 because there is only one point
            }
            else {
                scalar w = scalar(1) / scalar(3);
                Vector3 p = geometry[f_a] * w + geometry[f_b] * w + geometry[f_c] * w;
                v_face_ids[0] = geometry.size();
                geometry.push_back(p);
                
                existing_faces[face] = v_face_ids;
            }

            for (int f_id : v_face_ids) {
                ids[j] = f_id;
                ++j;
            }
        }

        for (int k = 0; k < 20; ++k) {
            topologies[Tetra20].push_back(ids[k]);
        }
    }
    topologies[Tetra].clear();
}

// convert data in an intial mesh to a target mesh
struct MeshMap {
    Element s_elem; // what is the initial element type
    Element t_elem; // what is the target element type
    Mesh::Geometry ref_geometry; // vertices position in reference element
    Mesh::Topology elem_topo; // topology of linear elem
    std::vector<int> v_elem; // in which element the vertices is valid


    MeshMap(Element _s_elem, Element _t_elem, 
            const Mesh::Geometry& _ref_geometry, 
            const Mesh::Topology& _elem_topo,
            const std::vector<int>& _v_elem) 
        : s_elem(_s_elem), t_elem(_t_elem), ref_geometry(_ref_geometry), elem_topo(_elem_topo), v_elem(_v_elem)
    { }

    template<typename T>
    std::vector<T> convert(Mesh* mesh, const std::vector<T>& vals) {
        int nb_vert = elem_nb_vertices(s_elem);
        FEM_Shape* shape = get_fem_shape(s_elem);
        int t_id = 0;
        std::vector<T> new_vals(ref_geometry.size());
        for (int i = 0; i < ref_geometry.size(); ++i) {
            t_id = v_elem[i];
            new_vals[i] = T();
            Vector3 p = ref_geometry[i];
            std::vector<scalar> weights = shape->build_shape(p.x, p.y, p.z);
            for (int j = 0; j < weights.size(); ++j) {
                new_vals[i] += vals[mesh->topologies()[s_elem][t_id * nb_vert + j]] * weights[j];
            }
        }
        delete shape;
        return new_vals;
    }

    void apply_to_mesh(Mesh* mesh) {
        mesh->geometry() = convert<Vector3>(mesh, mesh->geometry());
        mesh->topologies()[t_elem] = elem_topo;
        mesh->topologies()[s_elem].clear();
    }
};



void tetra_refine(
    MeshMap* map,
    Mesh::Geometry& ref_tetra_geometry,
    Mesh::Topology& ref_tetra_edges,
    std::vector<int>& t_ids) {

    int tetra_10_topo[32] = { 0,4,6,7, 1,5,4,8, 7,8,9,3, 2,6,5,9, 6,4,5,7, 7,4,5,8, 6,5,9,7, 7,8,5,9 };
    std::map<Face<2>, int> edges;
    Mesh::Topology new_tetra_topology;
    std::vector<Vector3> new_ref_tetra_geometry;

    Mesh::Topology e_topo(2);
    std::vector<int> ids(10);
    std::vector<Vector3> ids_geometry(10);
    std::vector<int> new_tid;

    int t_id = 0;
    for (int i = 0; i < map->elem_topo.size(); i += 4) {
        t_id = t_ids[i / 4];
        for (int j = 0; j < 4; ++j) {
            ids[j] = map->elem_topo[i + j];
            ids_geometry[j] = ref_tetra_geometry[i + j];
        }

        for (int j = 0; j < ref_tetra_edges.size(); j += 2) {
            e_topo[0] = map->elem_topo[i + ref_tetra_edges[j]];
            e_topo[1] = map->elem_topo[i + ref_tetra_edges[j + 1]];

            Vector3 pa = ref_tetra_geometry[i + ref_tetra_edges[j]];
            Vector3 pb = ref_tetra_geometry[i + ref_tetra_edges[j + 1]];
            Vector3 p = scalar(0.5) * (pa + pb);


            Face<2> e(e_topo);
            int id;
            // edge found in map
            if (edges.find(e) != edges.end()) {
                id = edges[e];
            }
            else {
                id = map->ref_geometry.size();
                map->ref_geometry.push_back(p);
                map->v_elem.push_back(t_id);
                edges[e] = id;
            }
            ids_geometry[4 + j / 2] = p;
            ids[4 + j / 2] = id;
        }

        for (int k = 0; k < 32; ++k) {
            new_tetra_topology.push_back(ids[tetra_10_topo[k]]);
            new_ref_tetra_geometry.push_back(ids_geometry[tetra_10_topo[k]]);
        }

        for (int k = 0; k < 8; ++k) {
            new_tid.push_back(t_id);
        }
    }

    map->elem_topo = new_tetra_topology;
    ref_tetra_geometry = new_ref_tetra_geometry;
    t_ids = new_tid;
}


/// Create a mapping of a Tetra mesh into linear Tetra (that can be refined)
MeshMap* tetra_to_linear(Mesh* mesh, Element elem, int subdivision) {
    if (elem != Tetra && elem != Tetra10 && elem != Tetra20) return nullptr;

    Mesh::Topology tetras = mesh->topologies()[elem];
    int nb_vert = elem_nb_vertices(elem);

    TetraConverter* tetra_converter = new TetraConverter();
    tetra_converter->init();
    Mesh::Topology ref_tetra_edges = tetra_converter->get_elem_topo_edges();
    Mesh::Geometry ref_tetra_geom = tetra_converter->geo_ref();

    int nb_tetra = tetras.size() / nb_vert;

    // rebuild the mesh as linear tetrahedron mesh but with only position in reference element
    std::vector<int> v_ids(mesh->geometry().size(), -1); // permit to check if vertices allready defined or not
    std::vector<int> t_ids(nb_tetra); // in which tetrahedron is defined each tetrahedron t_id = [0,nb_tetra-1]

    std::vector<int> v_tetra; // in which element the vertices is valid
    Mesh::Geometry ref_geometry; // vertices position in reference element

    Mesh::Geometry ref_tetra_geometry(nb_tetra * 4); // vertices position of all linear tetra (in ref element)
    Mesh::Topology tetra_topology(nb_tetra * 4); // topology of linear tetra
    int v_id = 0;
    int t_id = 0;
    for (int i = 0; i < tetras.size(); i += nb_vert) {
        t_id = i / nb_vert;
        t_ids[t_id] = t_id;
        for (int j = 0; j < 4; ++j) // we only needs the first 4 vertices
        {
            int k = t_id * 4 + j;
            ref_tetra_geometry[k] = ref_tetra_geom[j];
            int id = tetras[i + j];
            if (v_ids[id] == -1) {
                v_tetra.push_back(t_id);
                ref_geometry.push_back(ref_tetra_geom[j]);
                tetra_topology[k] = v_id;

                v_ids[id] = v_id;
                v_id++;
            }
            else {
                tetra_topology[i / nb_vert * 4 + j] = v_ids[id];
            }
        }
    }

    MeshMap* map = new MeshMap(elem, Tetra, ref_geometry, tetra_topology, v_tetra);

    //Subdivide
    for (int i = 0; i < subdivision; ++i) {
        tetra_refine(map, ref_tetra_geometry, ref_tetra_edges, t_ids);
    }

    return map;
}


