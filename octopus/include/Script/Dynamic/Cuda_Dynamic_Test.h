#pragma once
#include "Core/Base.h"
#include "GPU/GPU_PBD.h"
#include "Mesh/Mesh.h"
#include "Core/Component.h"
#include<vector> // for vector

int create_graph_color(Mesh::Topology& topology, Element element, int nb_vert, std::vector<int>& colors);

struct Cuda_Dynamic final : Component {
    explicit Cuda_Dynamic(const scalar density,const scalar young,const scalar poisson, const int iteration = 30) : _density(density), _young(young), _poisson(poisson), _iteration(iteration), _mesh(nullptr), _gpu_pbd(nullptr) {
    }

    void init() override;

    void update() override;

    ~Cuda_Dynamic() override {
        delete _gpu_pbd;
    }
private:
    std::map<Element, int> _nb_color;
    std::map<Element, std::vector<Color>> _display_colors;
    std::map<Element, std::vector<int>> _elem_colors;

    scalar _density;
    scalar _young, _poisson;
    int _iteration;

    Mesh* _mesh;
    GPU_PBD_FEM* _gpu_pbd;
};