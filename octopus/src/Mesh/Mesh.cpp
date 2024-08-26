#include "Mesh/Mesh.h"
#include <algorithm>

Mesh::Geometry Mesh::get_elem_vertices(const Element elem, const int eid) {
    const int nb = elem_nb_vertices(elem);
    Geometry geom(nb);
    for (int i = 0; i < nb; ++i) {
        geom[i] = _geometry[_topologies[elem][eid * nb + i]];
    }
    return geom;
}

Mesh::Topology Mesh::get_elem_indices(const Element elem, const int eid) {
    const int nb = elem_nb_vertices(elem);
    Topology topo(nb);
    for (int i = 0; i < nb; ++i) {
        topo[i] = _topologies[elem][eid * nb + i];
    }
    return topo;
}
