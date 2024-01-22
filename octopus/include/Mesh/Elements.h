#pragma once
enum Element
{
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

unsigned int elem_nb_vertices(Element elem) {
    switch (elem)
    {
        case Line:      return 2;
        case Triangle:  return 3;
        case Quad:      return 4;
        case Tetra:     return 4;
        case Pyramid:   return 5;
        case Prism:     return 6;
        case Hexa:      return 8;
        case Tetra10:   return 10;
        case Tetra20:   return 20;
        default:        return 0;
	}
}

char* element_name(Element elem) {
    switch (elem)
    {
    case Line:      return "Line";
    case Triangle:  return "Triangle";
    case Quad:      return "Quad";
    case Tetra:     return "Tetra";
    case Pyramid:   return "Pyramid";
    case Prism:     return "Prism";
    case Hexa:      return "Hexa";
    case Tetra10:   return "Tetra10";
    case Tetra20:   return "Tetra20";
    default:        return 0;
    }
}