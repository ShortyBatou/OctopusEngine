#include "Mesh/Generator/BeamGenerator.h"

Mesh *BeamMeshGenerator::build() {
    Mesh *mesh = new Mesh();
    build_geometry_grid(mesh->geometry());
    apply_transform(mesh->geometry());
    for (int z = 0; z < _subdivisions.z - 1; ++z)
        for (int y = 0; y < _subdivisions.y - 1; ++y)
            for (int x = 0; x < _subdivisions.x - 1; ++x) {
                int ids[8];
                this->get_cell_vertices_ids(x, y, z, ids);
                add_geometry_at_cell(x, y, z, mesh->geometry());
                build_topo_at_cell(ids, mesh->topologies());
            }
    return mesh;
}

void BeamMeshGenerator::build_geometry_grid(Mesh::Geometry &geometry) const {
    for (int z = 0; z < _subdivisions.z; ++z)
        for (int y = 0; y < _subdivisions.y; ++y)
            for (int x = 0; x < _subdivisions.x; ++x)
                geometry.emplace_back(  static_cast<scalar>(x) * _x_step,
                                        static_cast<scalar>(y) * _y_step,
                                        static_cast<scalar>(z) * _z_step);
}

void BeamMeshGenerator::get_cell_vertices_ids(const int x, const int y, const int z, int *ids) const {
    ids[0] = icoord_to_id(x, y, z);
    ids[1] = icoord_to_id(x + 1, y, z);
    ids[2] = icoord_to_id(x + 1, y, z + 1);
    ids[3] = icoord_to_id(x, y, z + 1);

    ids[4] = icoord_to_id(x, y + 1, z);
    ids[5] = icoord_to_id(x + 1, y + 1, z);
    ids[6] = icoord_to_id(x + 1, y + 1, z + 1);
    ids[7] = icoord_to_id(x, y + 1, z + 1);
}

int BeamMeshGenerator::icoord_to_id(const int x, const int y, const int z) const {
    return x + y * _subdivisions.x + z * _subdivisions.y * _subdivisions.x;
}

void HexaBeamGenerator::build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) {
    for (int i = 0; i < 8; ++i)
        topologies[Hexa].push_back(ids[i]);
}


void PrismBeamGenerator::build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) {
    static int prysms[12]{
        0, 1, 3, 4, 5, 7,
        1, 2, 3, 5, 6, 7
    };
    for (const int& prysm: prysms)
        topologies[Prism].push_back(ids[prysm]);
}

void PyramidBeamGenerator::add_geometry_at_cell(const int x, const int y, const int z, Mesh::Geometry &geometry) {
    _mid_id = static_cast<int>(geometry.size());
    const Vector3 v(static_cast<scalar>(x) * _x_step + _x_step * 0.5,
                    static_cast<scalar>(y) * _y_step + _y_step * 0.5,
                    static_cast<scalar>(z) * _z_step + _z_step * 0.5);
    geometry.push_back(v);
}

void PyramidBeamGenerator::build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) {
    static int pyramids[30]{
        3, 2, 1, 0, 8,
        0, 1, 5, 4, 8,
        1, 2, 6, 5, 8,
        2, 3, 7, 6, 8,
        3, 0, 4, 7, 8,
        4, 5, 6, 7, 8
    };
    for (const int& pyramid: pyramids) {
        int id;
        if (pyramid == 8) id = _mid_id;
        else id = ids[pyramid];
        topologies[Pyramid].push_back(id);
    }
}

void TetraBeamGenerator::build_topo_at_cell(int ids[8], std::map<Element, Mesh::Topology> &topologies) {
    //static unsigned tetras[24]{ 0,4,6,5, 0,4,7,6, 0,7,3,6, 2,0,3,6, 2,0,6,1, 6,0,5,1 }; int nb = 24;
    static int tetras[24]{0, 4, 6, 5, 3, 6, 2, 0, 0, 4, 7, 6, 3, 6, 0, 7, 2, 0, 6, 1, 6, 0, 5, 1};
    //static unsigned tetras[20]{ 1,6,5,4, 1,2,6,3, 0,1,4,3, 7,6,3,4, 1,3,6,4 }; int nb = 20;
    //static unsigned tetras[20]{ 0,3,2,7, 7,4,5,0, 7,6,2,5, 2,1,0,5, 7,0,5,0}; int nb = 20;
    for (const int& tetra: tetras)
        topologies[Tetra].push_back(ids[tetra]);
}


void subdive_tetra(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies) {
    const Mesh::Topology& tetras = topologies[Tetra];
    topologies[Tetra].clear();
    topologies[Tetra10].clear();

    using Edge = std::pair<int, int>;
    int tetra_10_topo[32] = {
        0, 4, 6, 7, 1, 5, 4, 8, 7, 8, 9, 3, 2, 6, 5, 9, 6, 4, 5, 7, 7, 4, 5, 8, 6, 5, 9, 7, 7, 8, 5, 9
    };
    Edge tetra_edges[6] = {Edge(0, 1), Edge(1, 2), Edge(0, 2), Edge(0, 3), Edge(1, 3), Edge(2, 3)};
    std::map<Edge, int> edges;
    for (int i = 0; i < tetras.size(); i += 4) {
        int ids[10];
        int j = 0;
        for (; j < 4; ++j) ids[j] = tetras[i + j];

        for (auto &[e_first, e_second]: tetra_edges) {
            Edge e;
            int e1 = ids[e_first];
            int e2 = ids[e_second];
            if (e1 > e2) std::swap(e1, e2);
            e.first = e1;
            e.second = e2;

            int id;
            // edge found in map
            if (edges.find(e) != edges.end()) {
                id = edges[e];
            } else {
                id = static_cast<int>(geometry.size());
                Vector3 pa(geometry[e1]);
                Vector3 pb(geometry[e2]);

                Vector3 p = 0.5f * (pa + pb);
                geometry.push_back(p);
                edges[e] = id;
            }
            ids[j] = id;
            ++j;
        }

        for (const int& k: tetra_10_topo) {
            topologies[Tetra].push_back(ids[k]);
        }
    }
}

void tetra4_to_tetra10(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies) {
    const Mesh::Topology &tetras = topologies[Tetra];

    topologies[Tetra10].clear();

    using Edge = std::pair<int, int>;
    Edge tetra_edges[6] = {Edge(0, 1), Edge(1, 2), Edge(0, 2), Edge(0, 3), Edge(1, 3), Edge(2, 3)};
    std::map<Edge, int> edges;
    for (int i = 0; i < tetras.size(); i += 4) {
        int ids[10];
        int j = 0;
        for (; j < 4; ++j) ids[j] = tetras[i + j];

        for (Edge &tet_e: tetra_edges) {

            int e1 = ids[tet_e.first];
            int e2 = ids[tet_e.second];
            if (e1 > e2) std::swap(e1, e2);
            Edge e(e1,e2);

            int id;
            // edge found in map
            if (edges.find(e) != edges.end()) {
                id = edges[e];
            } else {
                id = static_cast<int>(geometry.size());
                Vector3 pa(geometry[e1]);
                Vector3 pb(geometry[e2]);

                Vector3 p = static_cast<scalar>(0.5) * (pa + pb);
                geometry.push_back(p);
                edges[e] = id;
            }
            ids[j] = id;
            ++j;
        }

        for (int id: ids) {
            topologies[Tetra10].push_back(id);
        }
    }
    topologies[Tetra].clear();
}

void tetra4_to_tetra20(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies) {
    Mesh::Topology &tetras = topologies[Tetra];
    topologies[Tetra20].clear();

    std::vector<int> ids(20);

    TetraConverter tetra_converter;
    const Mesh::Topology &tetra_edges = tetra_converter.get_elem_topo_edges();
    const Mesh::Topology &tetra_faces = tetra_converter.get_elem_topo_triangle();

    std::map<Face<2>, std::vector<int> > existing_edges;
    std::map<Face<3>, std::vector<int> > existing_faces;

    std::vector<int> v_edge_ids(2);
    std::vector<int> v_face_ids(1);

    for (int i = 0; i < tetras.size(); i += 4) {
        int j = 0;
        for (; j < 4; ++j) ids[j] = tetras[i + j];

        // edges
        for (int k = 0; k < tetra_edges.size(); k += 2) {
            int e_a = ids[tetra_edges[k]];
            int e_b = ids[tetra_edges[k + 1]];
            Face<2> edge({e_a, e_b});

            auto it = existing_edges.find(edge);
            // edge found in map
            if (it != existing_edges.end()) {
                v_edge_ids = existing_edges[edge];

                if (edge.ids[0] != it->first.ids[0])
                    std::reverse(v_edge_ids.begin(), v_edge_ids.end());
            } else {
                for (int w = 0; w < 2; w++) {
                    scalar weight = static_cast<scalar>(w + 1) / 3.f;
                    Vector3 p = geometry[e_a] * (1.f - weight) + geometry[e_b] * weight;
                    v_edge_ids[w] = static_cast<int>(geometry.size());
                    geometry.push_back(p);
                }
                existing_edges[edge] = v_edge_ids;
            }

            for (int e_id: v_edge_ids) {
                ids[j] = e_id;
                ++j;
            }
        }

        //faces
        for (int k = 0; k < tetra_faces.size(); k += 3) {
            int f_a = ids[tetra_faces[k]];
            int f_b = ids[tetra_faces[k + 1]];
            int f_c = ids[tetra_faces[k + 2]];
            Face<3> face({f_a, f_b, f_c});
            // edge found in map
            if (existing_faces.find(face) != existing_faces.end()) {
                v_face_ids = existing_faces[face];
                // only works for P3 because there is only one point
            } else {
                scalar w = 1.f / 3.f;
                Vector3 p = geometry[f_a] * w + geometry[f_b] * w + geometry[f_c] * w;
                v_face_ids[0] = static_cast<int>(geometry.size());
                geometry.push_back(p);

                existing_faces[face] = v_face_ids;
            }

            for (int f_id: v_face_ids) {
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

void hexa_to_hexa27(Mesh::Geometry &geometry, std::map<Element, Mesh::Topology> &topologies) {
    Mesh::Topology &hexas = topologies[Hexa];
    topologies[Hexa27].clear();

    std::vector<int> ids(27);

    HexaConverter hexa_converter;
    const Mesh::Topology &hexa_edges = hexa_converter.get_elem_topo_edges();
    const Mesh::Topology &hexa_faces = hexa_converter.get_elem_topo_quad();

    std::map<Face<2>, int > existing_edges;
    std::map<Face<4>, int > existing_faces;

    ;


    for (int i = 0; i < hexas.size(); i += 8) {
        int j = 0;
        //corner
        for (; j < 8; ++j) ids[j] = hexas[i + j];

        // edges
        for (int k = 0; k < hexa_edges.size(); k += 2) {
            int e_a = ids[hexa_edges[k]];
            int e_b = ids[hexa_edges[k + 1]];
            Face<2> edge({e_a, e_b});
            auto it = existing_edges.find(edge);

            int id;
            // edge found in map
            if (it != existing_edges.end()) {
                id = existing_edges[edge];
            } else {
                id = static_cast<int>(geometry.size());
                Vector3 p = 0.5f * (geometry[e_a] + geometry[e_b]);
                geometry.push_back(p);
                existing_edges[edge] = id;
            }
            ids[j] = id;
            ++j;
        }

        //faces
        for (int k = 0; k < hexa_faces.size(); k += 4) {
            int f_a = ids[hexa_faces[k]];
            int f_b = ids[hexa_faces[k + 1]];
            int f_c = ids[hexa_faces[k + 2]];
            int f_d = ids[hexa_faces[k + 3]];
            Face<4> face({f_a, f_b, f_c, f_d});
            // edge found in map
            int fid;
            if (existing_faces.find(face) != existing_faces.end()) {
                fid = existing_faces[face];
                // only works for P3 because there is only one point
            } else {
                scalar w = 1.f / 4.f;
                Vector3 p = geometry[f_a] * w + geometry[f_b] * w + geometry[f_c] * w + geometry[f_d] * w;
                fid = static_cast<int>(geometry.size());
                geometry.push_back(p);

                existing_faces[face] = fid;
            }
            ids[j] = fid;
            ++j;
        }

        // volume
        Vector3 center = Unit3D::Zero();
        for (int k = 0; k < 8; k++) {
            center += geometry[ids[k]];
        }
        center /= 8.f;
        ids[j] = static_cast<int>(geometry.size());
        geometry.push_back(center);

        for (int k = 0; k < 27; ++k) {
            topologies[Hexa27].push_back(ids[k]);
        }
    }
    topologies[Hexa].clear();
}



template<typename T>
std::vector<T> MeshMap::convert(Mesh *mesh, const std::vector<T> &vals) {
    const int nb_vert = elem_nb_vertices(s_elem);
    const FEM_Shape *shape = get_fem_shape(s_elem);
    std::vector<T> new_vals(ref_geometry.size());
    for (int i = 0; i < ref_geometry.size(); ++i) {
        const int t_id = v_elem[i];
        new_vals[i] = T();
        const Vector3 p = ref_geometry[i];
        std::vector<scalar> weights = shape->build_shape(p.x, p.y, p.z);
        for (int j = 0; j < weights.size(); ++j) {
            new_vals[i] += vals[mesh->topologies()[s_elem][t_id * nb_vert + j]] * weights[j];
        }
    }
    delete shape;
    return new_vals;
}

void MeshMap::apply_to_mesh(Mesh *mesh) {
    mesh->geometry() = convert<Vector3>(mesh, mesh->geometry());
    mesh->topologies()[t_elem] = elem_topo;
    mesh->topologies()[s_elem].clear();
}


void tetra_refine(
    MeshMap *map,
    Mesh::Geometry &ref_tetra_geometry,
    Mesh::Topology &ref_tetra_edges,
    std::vector<int> &t_ids) {
    int tetra_10_topo[32] = {
        0, 4, 6, 7, 1, 5, 4, 8, 7, 8, 9, 3, 2, 6, 5, 9, 6, 4, 5, 7, 7, 4, 5, 8, 6, 5, 9, 7, 7, 8, 5, 9
    };
    std::map<Face<2>, int> edges;
    Mesh::Topology new_tetra_topology;
    std::vector<Vector3> new_ref_tetra_geometry;

    Mesh::Topology e_topo(2);
    std::vector<int> ids(10);
    std::vector<Vector3> ids_geometry(10);
    std::vector<int> new_tid;

    for (int i = 0; i < map->elem_topo.size(); i += 4) {
        int t_id = t_ids[i / 4];
        for (int j = 0; j < 4; ++j) {
            ids[j] = map->elem_topo[i + j];
            ids_geometry[j] = ref_tetra_geometry[i + j];
        }

        for (int j = 0; j < ref_tetra_edges.size(); j += 2) {
            e_topo[0] = map->elem_topo[i + ref_tetra_edges[j]];
            e_topo[1] = map->elem_topo[i + ref_tetra_edges[j + 1]];

            Vector3 pa = ref_tetra_geometry[i + ref_tetra_edges[j]];
            Vector3 pb = ref_tetra_geometry[i + ref_tetra_edges[j + 1]];
            Vector3 p = 0.5f * (pa + pb);

            Face<2> e(e_topo);
            int id;
            // edge found in map
            if (edges.find(e) != edges.end()) {
                id = edges[e];
            } else {
                id = static_cast<int>(map->ref_geometry.size());
                map->ref_geometry.push_back(p);
                map->v_elem.push_back(t_id);
                edges[e] = id;
            }
            ids_geometry[4 + j / 2] = p;
            ids[4 + j / 2] = id;
        }

        for (int k: tetra_10_topo) {
            new_tetra_topology.push_back(ids[k]);
            new_ref_tetra_geometry.push_back(ids_geometry[k]);
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
MeshMap *tetra_to_linear(Mesh *mesh, const Element elem, const int subdivision) {
    if (elem != Tetra && elem != Tetra10 && elem != Tetra20) return nullptr;

    const Mesh::Topology tetras = mesh->topologies()[elem];
    const int nb_vert = elem_nb_vertices(elem);

    TetraConverter *tetra_converter = new TetraConverter();
    tetra_converter->init();
    Mesh::Topology ref_tetra_edges = tetra_converter->get_elem_topo_edges();
    const Mesh::Geometry ref_tetra_geom = tetra_converter->geo_ref();

    const int nb_tetra = static_cast<int>(tetras.size()) / nb_vert;

    // rebuild the mesh as linear tetrahedron mesh but with only position in reference element
    std::vector<int> v_ids(mesh->geometry().size(), -1); // permit to check if vertices allready defined or not
    std::vector<int> t_ids(nb_tetra); // in which tetrahedron is defined each tetrahedron t_id = [0,nb_tetra-1]

    std::vector<int> v_tetra; // in which element the vertices is valid
    Mesh::Geometry ref_geometry; // vertices position in reference element

    Mesh::Geometry ref_tetra_geometry(nb_tetra * 4); // vertices position of all linear tetra (in ref element)
    Mesh::Topology tetra_topology(nb_tetra * 4); // topology of linear tetra

    int v_id = 0;
    for (int i = 0; i < tetras.size(); i += nb_vert) {
        int t_id = i / nb_vert;
        t_ids[t_id] = t_id;
        for (int j = 0; j < 4; ++j) // we only needs the first 4 vertices
        {
            const int k = t_id * 4 + j;
            ref_tetra_geometry[k] = ref_tetra_geom[j];
            int id = tetras[i + j];
            if (v_ids[id] == -1) {
                v_tetra.push_back(t_id);
                ref_geometry.push_back(ref_tetra_geom[j]);
                tetra_topology[k] = v_id;

                v_ids[id] = v_id;
                v_id++;
            } else {
                tetra_topology[i / nb_vert * 4 + j] = v_ids[id];
            }
        }
    }

    MeshMap *map = new MeshMap(elem, Tetra, ref_geometry, tetra_topology, v_tetra);

    //Subdivide
    for (int i = 0; i < subdivision; ++i) {
        tetra_refine(map, ref_tetra_geometry, ref_tetra_edges, t_ids);
    }

    return map;
}