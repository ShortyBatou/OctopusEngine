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
    Tetra20,
    Hexa27
};

// false if linear, else true
bool is_high_order(Element element);

// gives the linear version of element (ex: T10 => T4, Hexa => Hexa)
Element get_linear_element(Element element);

// gives element by the number of vertices (does not work with triangle, edge and quads)
Element get_elem_by_size(int nb);

// gives the numver of vertices in the element
int elem_nb_vertices(Element element);

// gives the element name
std::string element_name(Element element);

// !!! Only works with linear elements !!!
// gives the triangles / quads or edges of an element.
// The topology is in the reference element.
std::vector<int> ref_triangles(Element element);
std::vector<int> ref_quads(Element element);
std::vector<int> ref_edges(Element element);