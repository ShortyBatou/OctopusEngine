#pragma once
enum Element
{
	Line, 
	Triangle, 
	Quad, 
	Tetra, 
	Pyramid,
	Prysm, 
	Hexa, 
	Tetra10,
};

unsigned int element_vertices(Element elem) {
    switch (elem)
    {
        case Line:      return 2;
        case Triangle:  return 3;
        case Quad:      return 4;
        case Tetra:     return 4;
        case Pyramid:   return 5;
        case Prysm:     return 6;
        case Hexa:      return 8;
        case Tetra10:   return 10;
        default:        return 0;
	}
}