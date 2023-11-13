#pragma once
#include <map>
#include <vector>

#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Mesh/Elements.h"
template<unsigned int nb>
struct Face
{
    std::vector<unsigned int> _ids;
    std::vector<unsigned int> _ids_sorted;
    Face(std::vector<unsigned int> ids)
    {
        assert(ids.size() == nb);
        _ids = ids; 
        _ids_sorted = _ids;
        std::sort(_ids_sorted.begin(), _ids_sorted.end());
    }
    bool operator<(const Face<nb>& f) const
    {
        return _ids_sorted < f._ids_sorted;
    }

    /// \returns true if the other face \p f has the same vertice indices in the
    /// same order modulo rotation.
    bool operator==(const Face& f) const
    {
        for (unsigned int i = 0; i < nb; ++i)
            if (_ids_sorted[i] != f._ids_sorted[i]) return false;
        
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
    using Topology = std::vector<unsigned int>;
    using Geometry = std::vector<Vector3>;

    Mesh(bool dynamic_geometry = false, bool dynamic_topology = false)
        : _dynamic_geometry(dynamic_geometry)
        , _dynamic_topology(dynamic_topology)
        , _need_update(true)
	{ }
   
    Vector3& vertice(unsigned int i) { return _geometry[i]; }
    Geometry& geometry() { return _geometry; }
    unsigned int nb_vertices() { return _geometry.size(); }
    Topology& topology(Element elem) {return _topologies[elem]; }
    std::map<Element, Topology>& topologies() { return _topologies; }

    void setGeometry(const Geometry& geometry) {_geometry = geometry;}
    void setTopology(Element elem, const Topology& topology) { _topologies[elem] = topology;}
    void clear() { 
        _topologies.clear();
        _geometry.clear();
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