#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_Dynamic_Test.h"
#include "Tools/Random.h"
#include "GPU/GPU_PBD.h"
#include <random>
#include <algorithm>
#include <set>

int create_graph_color(Mesh::Topology& topology, Element element, int nb_vert, std::vector<int>& colors) {
    int nb_color = 1;
    int elem_nb_vert = elem_nb_vertices(element);
    std::vector<std::set<int>> owners(nb_vert);
    // for each vertice get elements that own this vertice
    for(int i = 0; i < topology.size(); i+=elem_nb_vert) {
        for(int j = 0; j < elem_nb_vert; ++j) {
            owners[topology[i+j]].insert(i/elem_nb_vert);
        }
    }

    colors.resize(topology.size() / elem_nb_vert, -1);
    std::vector<int> available(64, true);
    for(int i = 0; i < topology.size(); i+=elem_nb_vert) {
        // for all vertices, check the neighbor elements colors
        for(int j = 0; j < elem_nb_vert; ++j) {
            for(int n : owners[topology[i+j]]) {
                if(colors[n] != -1) available[colors[n]] = false;
            }
        }
        for(int c = 0; c < available.size(); ++c) {
            if(available[c]) {
                nb_color = std::max(nb_color, c);
                colors[i/elem_nb_vert] = c; break;
            }
        }
        std::fill(available.begin(), available.end(), true);
    }
    std::cout << "NB color: " << nb_color << std::endl;
    return nb_color;
}

void Cuda_Dynamic::init() {
    _mesh = _entity->get_component<Mesh>();
    for(auto& it : _mesh->topologies()) {
        Element e = it.first;
        const int elem_nb_vert = elem_nb_vertices(e);
        const int nb_element = static_cast<int>(it.second.size())/elem_nb_vert;
        _nb_color[e] = create_graph_color(it.second, e, static_cast<int>(_mesh->geometry().size()), _elem_colors[e]);
        // sort element by color and get color group sizes
        int count = 0;
        std::vector<int> offsets(_nb_color[e],0);
        std::vector<int> sorted_topology(it.second.size());
        for(int c = 0; c < _nb_color[e]; ++c) {
            offsets[c] = count;
            for(int i = 0; i < nb_element; ++i) {
                if(_elem_colors[e][i] != c) continue;
                int id = i * elem_nb_vert;
                sorted_topology.insert(sorted_topology.begin() + count, it.second.begin() + id, it.second.begin() + id + elem_nb_vert);
                count += elem_nb_vert;
            }
        }
        // create CUDA PBD Gauss-Seidel
        _gpu_pbd = new GPU_PBD_FEM(e, _mesh->geometry(), sorted_topology, offsets, _density);

    }
}

void Cuda_Dynamic::update() {
    if(Time::Frame() == 1) {
        // coloration
        for(auto& it : _mesh->topologies()) {
            Element e = it.first;
            std::vector<Color> color_map(_nb_color[e]+1);
            for(int i = 0 ; i < color_map.size(); ++i) {
                color_map[i] = ColorBase::HSL2RGB(Random::Range(0.f,360.f), Random::Range(42.f,98.f), Random::Range(40.f,90.f));
            }
            _display_colors[e].resize(_elem_colors[e].size());
            for(int i = 0; i < _elem_colors[e].size(); ++i) {
                _display_colors[e][i] = color_map[_elem_colors[e][i]];
            }
        }
    }

    if(Time::Frame() > 0) {
        GL_Graphic* graphic = entity()->get_component<GL_Graphic>();
        for(auto& it : _mesh->topologies()) {
            graphic->set_ecolors(it.first, _display_colors[it.first]);
            graphic->set_multi_color(true);
            graphic->set_element_color(true);
        }

    }

    _gpu_pbd->step(Time::Fixed_DeltaTime());
    _gpu_pbd->get_position(_mesh->geometry());
}

