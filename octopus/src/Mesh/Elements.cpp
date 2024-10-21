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

int elem_nb_vertices(const Element element) {
    switch (element) {
        case Line: return 2;
        case Triangle: return 3;

        case Quad:
        case Tetra: return 4;

        case Pyramid: return 5;
        case Prism: return 6;
        case Hexa: return 8;
        case Tetra10: return 10;
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
