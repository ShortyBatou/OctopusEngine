#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"
class GL_GraphicElement : public GL_Graphic
{
public:
    GL_GraphicElement(scalar scale = 0.9) : GL_Graphic(Color(0.9, 0.3, 0.3, 1.0)), _scale(scale)
    {
        _converters[Tetra]    = new TetraConverter();
        _converters[Pyramid]  = new PyramidConverter();
        _converters[Prism]    = new PrysmConverter();
        _converters[Hexa]     = new HexaConverter();
        _converters[Tetra10]  = new Tetra10Converter();
        _converters[Tetra20] = new Tetra20Converter();
        for (auto& elem : _converters) elem.second->init();
        this->_multi_color = true;
    }

    virtual void get_geometry(Mesh::Geometry& geometry) override
    {
        for (const auto& elem : this->_mesh->topologies())
        {
            Element type = elem.first;
            if (_converters.find(type) == _converters.end()) continue;
            _converters[type]->build_scaled_geometry(this->_mesh->geometry(), this->_mesh->topologies(), geometry, _scale);
        }

    }

    virtual void get_topology(
        Mesh::Topology& lines, 
        Mesh::Topology& triangles, 
        Mesh::Topology& quads, 
        Mesh::Topology& tri_to_elem, 
        Mesh::Topology& quad_to_elem) override
    {
        for (const auto& elem : this->_mesh->topologies())
        {
            Element type = elem.first;
            if (_converters.find(type) == _converters.end()) continue;
            _converters[type]->build_scaled_topology(this->_mesh->topologies(), lines, triangles, quads, tri_to_elem, quad_to_elem);
        }
    }

    virtual void update_buffer_colors() override
    {
        this->_colors.clear();
        for (const auto& elem : this->_mesh->topologies())
        {
            Element type = elem.first;
            unsigned int nb_vertices = elem.second.size();
            std::vector<Color> elem_colors(nb_vertices, element_colors[type]);
            this->_colors.insert(this->_colors.end(), elem_colors.begin(), elem_colors.end());
        }
        _b_color->load_data(_colors);
    }

    static std::map<Element, Color> element_colors;

protected:
    scalar _scale;
    std::map<Element, MeshConverter*> _converters;
};

std::map<Element, Color> GL_GraphicElement::element_colors = {
    {Line, ColorBase::Red()},
    {Triangle, ColorBase::Blue()},
    {Quad, ColorBase::Green()},
    {Tetra, Color(0.9, 0.3, 0.3, 1.0)},
    {Pyramid, Color(0.9, 0.5, 0.1, 1.0)},
    {Prism, Color(0.3, 0.9, 0.5, 1.0)},
    {Hexa, Color(0.3, 0.9, 0.3, 1.0)},
    {Tetra10, Color(0.3, 0.3, 0.9, 1.0)},
    {Tetra20, Color(0.3, 0.7, 0.9, 1.0)}
};