#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"
#include <map>
#include <set>

class GL_GraphicSurface : public GL_Graphic
{
public:
    GL_GraphicSurface(const Color& color = Color(0.8, 0.4, 0.4, 1.0)) : GL_Graphic(color)
    {
        _converters[Line]     = new LineConverter();
        _converters[Triangle] = new TriangleConverter();
        _converters[Quad]     = new QuadConverter();
        _converters[Tetra]    = new TetraConverter();
        _converters[Pyramid]  = new PyramidConverter();
        _converters[Prysm]    = new PrysmConverter();
        _converters[Hexa]     = new HexaConverter();
        for (auto& elem : _converters) elem.second->init();
    }

    // find the surface of the mesh (pretty much brute force maybe, there is a better way)
    virtual void update_buffer_topology() override
    {
        std::map<Element, Mesh::Topology> elem_topologies;
        elem_topologies[Line] = Mesh::Topology();
        elem_topologies[Triangle] = Mesh::Topology();
        elem_topologies[Quad]     = Mesh::Topology();
        for (const auto& elem : this->_mesh->topologies())
        {
            Element type = elem.first;
            if (_converters.find(type) == _converters.end()) continue;

            // convert all elements into triangles (quad are cuted in 2 triangles)
            _converters[type]->convert_element( this->_mesh->topologies(), elem_topologies);
        }

        // revome duplicate faces
        get_surface<3>(elem_topologies[Triangle]);
        get_surface<6>(elem_topologies[Quad]); // quad = 2 triangle here

        std::vector<unsigned int> ids(2);
        for (unsigned int i = 0; i < elem_topologies[Quad].size(); i += 6)
        {
            // find the common edge between the 2 quad's triangles
            Face<2> edge = find_edge(elem_topologies[Quad], i);
            for (unsigned int j = 0; j < 3; ++j)
            for (unsigned int k = 0; k < 4; k += 3)
            {
                ids[0] = elem_topologies[Quad][i + j + k];
                ids[1] = elem_topologies[Quad][i + (j + 1) % 3 + k];
                Face<2> line(ids);
                // if the line is the common edge, ignore. Else, add it
                if (line != edge)
                    elem_topologies[Line].insert(
                        elem_topologies[Line].begin(), line._ids.begin(),
                        line._ids.end());
            }
        }

        if (elem_topologies[Line].size() > 0)
            this->_b_line->load_data(elem_topologies[Line]);
        if (elem_topologies[Triangle].size() > 0)
            this->_b_triangle->load_data(elem_topologies[Triangle]);
        if (elem_topologies[Quad].size() > 0)
            this->_b_quad->load_data(elem_topologies[Quad]);
    }

protected:

    template<unsigned int nb>
    void get_surface(Mesh::Topology& topology, bool remove_double = true)
    {
        std::set<Face<nb>> faces;
        Mesh::Topology ids(nb);
        for (unsigned int i = 0; i < topology.size(); i += nb)
        {
            for (unsigned int j = 0; j < nb; ++j) ids[j] = topology[i + j];
            Face<nb> face(ids);
            auto it = faces.find(face);
            if (it == faces.end()) 
                faces.insert(face);
            else if(remove_double)
                faces.erase(it);
        }
        topology.clear();
        for (auto& face : faces)
            topology.insert(topology.end(), face._ids.begin(), face._ids.end());
    }

    Face<2> find_edge(Mesh::Topology& topology, unsigned int i_start)
    {
        std::vector<unsigned int> ids;
        for (unsigned int j = 0; j < 3; ++j)
        {
            unsigned int current = topology[i_start + j];
            for (unsigned int k = 0; k < 3; ++k)
            {
                if (current == topology[i_start + k + 3])
                    ids.push_back(current);
            }
        }
         return Face<2>(ids);
    }

    std::map<Element, MeshConverter*> _converters;
};