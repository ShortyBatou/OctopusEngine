#include "Mesh/Elements.h"

bool is_high_order(const Element element) {
    switch (element) {
        // linear
        case Line:
        case Triangle:
        case Quad:
        case Tetra:
        case Pyramid:
        case Prism:
        case Hexa: return false;

        // high-order
        case Tetra10:
        case Tetra20:
        case Hexa27: return true;

        default: return false;
    }
}

Element get_elem_by_size(const int nb) {
    switch (nb) {
        case 4: return Tetra;
        case 5: return Pyramid;
        case 6: return Prism;
        case 8: return Hexa;
        case 10: return Tetra10;
        case 20: return Tetra20;
        case 27: return Hexa27;
        default: return Line;
    }
}

Element get_linear_element(Element element) {
    switch (element) {
        case Line:
        case Triangle:
        case Quad:
        case Tetra:
        case Pyramid:
        case Prism:
        case Hexa: return element;
        case Tetra10:  return Tetra;
        case Tetra20: return Tetra;
        case Hexa27: return Hexa;
        default: return Line;
    }
}

int elem_nb_vertices(const Element element) {
    switch (element) {
        case Line: return 2;
        case Triangle: return 3;

        case Quad:
        case Tetra: return 4;

        case Pyramid: return 5;
        case Prism: return 6;
        case Hexa: return 8;
        case Tetra10:  return 10;
        case Tetra20: return 20;
        case Hexa27: return 27;
        default: return 0;
    }
}

std::string element_name(const Element element) {
    switch (element) {
        case Line: return "Line";
        case Triangle: return "Triangle";
        case Quad: return "Quad";
        case Tetra: return "Tetra";
        case Pyramid: return "Pyramid";
        case Prism: return "Prism";
        case Hexa: return "Hexa";
        case Tetra10: return "Tetra10";
        case Tetra20: return "Tetra20";
        case Hexa27: return "Hexa27";
        default: return "";
    }
}

std::vector<int> ref_triangles(Element element) {
    switch (element) {
        case Line: {};
        case Triangle: return {0,1,2};
        case Quad: return {0,1,2,1,3,2};
        case Tetra: return {0, 1, 3, 1, 2, 3, 0, 3, 2, 0, 2, 1};
        case Pyramid: return {0, 1, 4, 3, 0, 4, 1, 2, 4, 2, 3, 4};
        case Prism: return {0, 1, 2, 3, 5, 4};
        default: return {};
    }
}

std::vector<int> ref_quads(Element element) {
    switch (element) {
        case Quad: return {0,1,2,3};
        case Pyramid: return {0,1,2,3};
        case Prism: return {3, 4, 1, 0, 2, 5, 3, 0, 1, 4, 5, 2};
        case Hexa: return {0, 1, 2, 3, 4, 5, 1, 0, 5, 6, 2, 1,3, 2, 6, 7,7, 4, 0, 3,7, 6, 5, 4};
        default: return {};
    }
}

std::vector<int> ref_edges(Element element) {
    switch (element) {
        case Line: return {0,1};
        case Triangle: return {0,1,1,2,2,0};
        case Quad: return {0,1,1,3,3,2,2,0};
        case Tetra: return {0, 1, 1, 2, 2, 0, 0, 3, 1, 3, 2, 3};
        case Pyramid: return {0, 1, 1, 2, 2, 3, 0, 4, 1, 4, 2, 4, 3, 4};
        case Prism: return {0, 1, 1, 2, 2, 0, 0, 3, 1, 4, 2, 5, 3, 4, 4, 5, 5, 3};
        case Hexa: return {0, 1, 1, 2, 2, 3, 3, 0, 0, 4, 1, 5, 2, 6, 3, 7, 4, 5, 5, 6, 6, 7, 7, 4};
        default: return {};
    }
}