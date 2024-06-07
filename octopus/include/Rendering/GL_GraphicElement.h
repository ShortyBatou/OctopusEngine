#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"

class GL_GraphicElement : public GL_Graphic
{
public:
    GL_GraphicElement(Color color, scalar scale = 0.9) : GL_Graphic(color), _scale(scale)
    {
        _converters[Tetra]    = new TetraConverter();
        _converters[Pyramid]  = new PyramidConverter();
        _converters[Prism]    = new PrysmConverter();
        _converters[Hexa]     = new HexaConverter();
        _converters[Tetra10]  = new Tetra10Converter();
        _converters[Tetra20] = new Tetra20Converter();
        for (auto& elem : _converters) elem.second->init();
    }

    virtual void update_gl_geometry() override
    { 
        for (const auto& elem : _mesh->topologies())
        {
            Element type = elem.first;
            if (_converters.find(type) == _converters.end()) continue;
            _converters[type]->build_scaled_geometry(_mesh->geometry(), _mesh->topologies(), _gl_geometry->geometry, _scale);
        }
    }

    virtual void update_gl_topology() override
    {
        for (const auto& it : _mesh->topologies())
        {
            Element element = it.first;
            if (_converters.find(element) == _converters.end()) continue;
            GL_Topology* gl_topo = _gl_topologies[element];
            Mesh::Topology quads, triangles;
            _converters[element]->build_scaled_topology(_mesh->topology(element), triangles, quads);

            const Mesh::Topology& ref_topology_tri = _converters[element]->topo_triangle();
            const Mesh::Topology& ref_topology_quad = _converters[element]->topo_quad();

            // convert element to triangles and quads (each element are displayed separatly)
            Mesh::Topology tri_to_elem, quad_to_elem;
            MeshTools::ExtractTopoToElem<3>(triangles, ref_topology_tri, tri_to_elem);
            MeshTools::ExtractTopoToElem<4>(quads, ref_topology_quad, quad_to_elem);
            
            // convert quads into two triangles
            static int quad_lines[8] = { 0,1,1,2,2,3,3,0 };
            int quad_triangle[6] = { 0,1,3, 3,1,2 };
            gl_topo->quads.resize(quads.size() / 4 * 6);
            gl_topo->quad_to_elem.resize(quads.size() / 4 * 2);
            for (int i = 0; i < quads.size() / 4; i++) {
                for (int j = 0; j < 6; ++j) {
                    gl_topo->quads[i * 6 + j] = quads[i * 4 + quad_triangle[j]];
                }

                gl_topo->quad_to_elem[i * 2] = quad_to_elem[i];
                gl_topo->quad_to_elem[i * 2 + 1] = quad_to_elem[i];
            }

            // Get wireframe
            if (!is_high_order(element)) {
                gl_topo->triangles = triangles;
                gl_topo->tri_to_elem = tri_to_elem;
                gl_topo->lines.resize(quads.size() / 4 * 8);
                for (int i = 0; i < quads.size() / 4; i++) {
                    for (int j = 0; j < 8; ++j) {
                        gl_topo->lines[i * 8 + j] = quads[i * 4 + quad_lines[j]];
                    }
                }
            }
            else {
                gl_topo->quads.insert(gl_topo->quads.end(), triangles.begin(), triangles.end());
                gl_topo->quad_to_elem.insert(gl_topo->quad_to_elem.end(), tri_to_elem.begin(), tri_to_elem.end());
                Mesh::Topology lines;
                _converters[element]->get_scaled_wireframe(_mesh->topology(element), lines);
                gl_topo->lines = lines;
            }
        }
    }

    virtual void update_gl_vcolors()
    {
        _gl_geometry->vcolors.clear();
        for (const auto& it : _mesh->topologies())
        {
            Element element = it.first;
            int size = _gl_geometry->vcolors.size();
            _gl_geometry->vcolors.resize(_gl_geometry->vcolors.size() + it.second.size());
            const Mesh::Topology& topology = it.second;

            for (int i = 0; i < topology.size(); ++i) {
                _gl_geometry->vcolors[size + i] = _vcolors[topology[i]];
            }
        }
    }
protected:
    scalar _scale;
    std::map<Element, MeshConverter*> _converters;
};