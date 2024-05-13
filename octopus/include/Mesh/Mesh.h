#pragma once
#include <map>
#include <vector>
#include <algorithm>
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Mesh/Elements.h"
template<int nb>
struct Face
{
    using Geometry = std::vector<Vector3>;
    std::vector<int> ids;
    std::vector<int> ids_sorted;
    Geometry vertices;

    int element_id;
    int face_id;

    Face(const std::vector<int>& _ids, 
        const Geometry& _vertices = {},
        int _element_id = 0, 
        int _face_id = 0) : element_id(_element_id), face_id(_face_id), vertices(_vertices)
    {
        assert(_ids.size() == nb);
        ids = _ids; 
        ids_sorted = _ids;
        std::sort(ids_sorted.begin(), ids_sorted.end());
    }

    bool operator<(const Face<nb>& f) const
    {
        return ids_sorted < f.ids_sorted;
    }

    /// \returns true if the other face \p f has the same vertice indices in the
    /// same order modulo rotation.
    bool operator==(const Face& f) const
    {
        for (int i = 0; i < nb; ++i)
            if (ids_sorted[i] != f.ids_sorted[i]) return false;
        
        return true;
    }

    bool operator!=(const Face& f) const
    {
        if (*this == f) return false;
        return true;
    }
};

class Mesh : public Behaviour
{
public:
    using Topology = std::vector<int>;
    using Geometry = std::vector<Vector3>;

    Mesh(bool dynamic_geometry = false, bool dynamic_topology = false)
        : _dynamic_geometry(dynamic_geometry)
        , _dynamic_topology(dynamic_topology)
        , _need_update(true)
	{ }
    
    Vector3& vertice(int i) { return _geometry[i]; }
    Geometry& geometry() { return _geometry; }
    int nb_vertices() { return _geometry.size(); }
    Topology& topology(Element elem) {return _topologies[elem]; }
    std::map<Element, Topology>& topologies() { return _topologies; }

    void set_geometry(const Geometry& geometry) {_geometry = geometry;}
    void set_topology(Element elem, const Topology& topology) { _topologies[elem] = topology;}
    void clear() { 
        _topologies.clear();
        _geometry.clear();
    }

    Geometry get_elem_vertices(Element elem, int eid) {
        int nb = elem_nb_vertices(elem);
        Geometry geom(nb);
        for (int i = 0; i < nb; ++i) {
            geom[i] = _geometry[_topologies[elem][eid * nb + i]];
        }
        return geom;
    }

    Topology get_elem_indices(Element elem, int eid) {
        int nb = elem_nb_vertices(elem);
        Topology topo(nb);
        for (int i = 0; i < nb; ++i) {
            topo[i] = _topologies[elem][eid * nb + i];
        }
        return topo;
    }

    void update_mesh() { _need_update = true; }
    void set_dynamic_geometry(bool state) { _dynamic_geometry = state; }
    void set_dynamic_topology(bool state) { _dynamic_topology = state; }
    bool& need_update() { return _need_update; }
    bool has_dynamic_geometry() const { return _dynamic_geometry; }
    bool has_dynamic_topology() const { return _dynamic_topology; }

protected:
    std::map<Element, Topology> _topologies;
	std::vector<Vector3> _geometry;
	// need_update : force update (even if static)
    bool _need_update, _dynamic_geometry, _dynamic_topology;
};
