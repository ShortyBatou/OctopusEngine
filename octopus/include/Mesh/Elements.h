#pragma once
#include <string>

enum Element {
    Line,
    Triangle,
    Quad,
    Tetra,
    Pyramid,
    Prism,
    Hexa,
    Tetra10,
    Tetra20
};

bool is_high_order(Element element);

int elem_nb_vertices(Element element);

std::string element_name(Element element);
