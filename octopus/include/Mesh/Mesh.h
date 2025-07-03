#pragma once
#include <map>
#include <vector>
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Mesh/Elements.h"
#include <algorithm>

template<int nb>
struct Face {
    using Geometry = std::vector<Vector3>;
    std::vector<int> ids;
    std::vector<int> ids_sorted;
    Geometry vertices;

    int element_id;
    int face_id;

    explicit Face(const std::vector<int> &_ids,
                  const Geometry &_vertices = {},
                  int _element_id = 0,
                  int _face_id = 0) : element_id(_element_id), face_id(_face_id), vertices(_vertices) {
        assert(_ids.size() == nb);
        build_ids(_ids);
    }

    bool operator<(const Face &f) const;

    /// \returns true if the other face \p f has the same vertice indices in the
    /// same order modulo rotation.
    bool operator==(const Face &f) const;

    bool operator!=(const Face &f) const;

    void build_ids(const std::vector<int> &_ids);
};

class Mesh : public Behaviour {
public:
    using Topology = std::vector<int>;
    using Geometry = std::vector<Vector3>;

    explicit Mesh(bool dynamic_geometry = false, bool dynamic_topology = false)
        : _dynamic_geometry(dynamic_geometry)
          , _dynamic_topology(dynamic_topology)
          , _need_update(true) {
    }

    Vector3 &vertice(int i) { return _geometry[i]; }
    Geometry &geometry() { return _geometry; }
    [[nodiscard]] int nb_vertices() const { return static_cast<int>(_geometry.size()); }
    Topology &topology(Element elem) { return _topologies[elem]; }
    std::map<Element, Topology> &topologies() { return _topologies; }

    void set_geometry(const Geometry &geometry) { _geometry = geometry; }
    void set_topology(Element elem, const Topology &topology) { _topologies[elem] = topology; }

    void clear() {
        _topologies.clear();
        _geometry.clear();
    }

    Geometry get_elem_vertices(Element elem, int eid);

    Topology get_elem_indices(Element elem, int eid);

    void update_mesh() { _need_update = true; }
    void set_dynamic_geometry(bool state) { _dynamic_geometry = state; }

    void set_dynamic_topology(bool state) {
        _dynamic_topology = state;
    }
    bool has_element_type(const Element type) { return _topologies[type].size() > 0;}
    bool &need_update() { return _need_update; }
    [[nodiscard]] bool has_dynamic_geometry() const { return _dynamic_geometry; }
    [[nodiscard]] bool has_dynamic_topology() const { return _dynamic_topology; }

protected:
    std::map<Element, Topology> _topologies;
    std::vector<Vector3> _geometry;
    // need_update : force update (even if static)
    bool _need_update, _dynamic_geometry, _dynamic_topology;
};

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
void Face<nb>::build_ids(const std::vector<int> &_ids)
{
    ids = _ids;
    ids_sorted = _ids;
    std::sort(ids_sorted.begin(), ids_sorted.end());
}