#include "Mesh/Mesh.h"
#include <algorithm>

template<int nb>
bool Face<nb>::operator<(const Face &f) const {
    return ids_sorted < f.ids_sorted;
}

template<int nb>
bool Face<nb>::operator==(const Face &f) const {
    for (int i = 0; i < nb; ++i)
        if (ids_sorted[i] != f.ids_sorted[i]) return false;

    return true;
}

template<int nb>
bool Face<nb>::operator!=(const Face &f) const {
    if (*this == f) return false;
    return true;
}

template<int nb>
void Face<nb>::build_ids(const std::vector<int> &_ids) {
    ids = _ids;
    ids_sorted = _ids;
    std::sort(ids_sorted.begin(), ids_sorted.end());
}

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
