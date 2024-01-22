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
        _converters[Tetra]    = new TetraConverter();
        _converters[Pyramid]  = new PyramidConverter();
        _converters[Prism]    = new PrysmConverter();
        _converters[Hexa]     = new HexaConverter();
        _converters[Tetra10]  = new Tetra10Converter();
        _converters[Tetra20] = new Tetra20Converter();
        for (auto& elem : _converters) elem.second->init();
    }

    // find the surface of the mesh (pretty much brute force maybe, there is a better way)
    virtual void get_topology(Mesh::Topology& lines, Mesh::Topology& triangles, Mesh::Topology& quads) override
    {
        Mesh::Topology element_quads;
        for (const auto& elem : this->_mesh->topologies())
        {
            Element type = elem.first;
            if (_converters.find(type) == _converters.end()) continue;

            // convert all elements into triangles (quad are cuted in 2 triangles)
            _converters[type]->convert_element(this->_mesh->topologies(), triangles, element_quads);
        }

        // revome duplicate faces
        get_surface<3>(triangles);
        get_surface<4>(element_quads); // quad = 2 triangle here

        lines.resize(element_quads.size() / 4 * 8);
        quads.resize(element_quads.size() / 4 * 6);

        unsigned int quad_lines[8] = { 0,1,1,2,2,3,3,0 };
        unsigned int quad_triangle[6] = { 0,1,3, 3,1,2 };

        for (unsigned int i = 0; i < element_quads.size()/4; i ++)
        {
            for (unsigned int j = 0; j < 8; ++j) 
                lines[i*8+j] = element_quads[i*4 + quad_lines[j]];

            for (unsigned int j = 0; j < 6; ++j)
                quads[i * 6 + j] = element_quads[i * 4 + quad_triangle[j]];
        }

        get_surface<2>(lines, false); 
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
            topology.insert(topology.end(), face.ids.begin(), face.ids.end());
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