#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"
class GL_GraphicElement : public GL_Graphic
{
public:
    GL_GraphicElement() : GL_Graphic(Color(0.9, 0.3, 0.3, 1.0))
    {
        _converters[Line]     = new LineConverter();
        _converters[Triangle] = new TriangleConverter();
        _converters[Quad]     = new QuadConverter();
        _converters[Tetra]    = new TetraConverter();
        _converters[Pyramid]  = new PyramidConverter();
        _converters[Prysm]    = new PrysmConverter();
        _converters[Hexa]     = new HexaConverter();
        for (auto& elem : _converters) elem.second->init();
        this->_multi_color = true;
    }

    virtual void update_buffer_geometry() override
    {
        Mesh::Geometry elem_geometry;
        for (const auto& elem : this->_mesh->topologies())
        {
            Element type = elem.first;
            if (_converters.find(type) == _converters.end()) continue;
            _converters[type]->build_scaled_geometry(this->_mesh->geometry(),
                                            this->_mesh->topologies(),
                                            elem_geometry, 0.7f);
        }

        this->_b_vertex->load_data(elem_geometry);
    }

    virtual void update_buffer_topology() override
    {
        std::map<Element, Mesh::Topology> elem_topologies;
        elem_topologies[Line]  = Mesh::Topology();
        elem_topologies[Triangle] = Mesh::Topology();
        elem_topologies[Quad] = Mesh::Topology();
        for (const auto& elem : this->_mesh->topologies())
        {
            Element type = elem.first;
            if (_converters.find(type) == _converters.end()) continue;
            _converters[type]->build_scaled_topology(this->_mesh->topologies(),
                                             elem_topologies);
        }
        if (elem_topologies[Line].size() > 0)
            this->_b_line->load_data(elem_topologies[Line]);
        if (elem_topologies[Triangle].size() > 0)
            this->_b_triangle->load_data(elem_topologies[Triangle]);
        if (elem_topologies[Quad].size() > 0)
            this->_b_quad->load_data(elem_topologies[Quad]);
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
    std::map<Element, MeshConverter*> _converters;
};

std::map<Element, Color> GL_GraphicElement::element_colors = {
    {Line, ColorBase::Red()},
    {Triangle, ColorBase::Blue()},
    {Quad, ColorBase::Green()},
    {Tetra, Color(0.3, 0.3, 0.9, 1.0)},
    {Pyramid, Color(0.9, 0.5, 0.1, 1.0)},
    {Prysm, Color(0.9, 0.3, 0.3, 1.0)},
    {Hexa, Color(0.3, 0.9, 0.3, 1.0)}
};