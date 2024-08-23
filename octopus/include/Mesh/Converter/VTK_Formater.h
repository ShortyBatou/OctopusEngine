#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include <iostream>
#include <fstream>
#include <string>


struct VTK {
	// only UNSTRUCTURED_GRID are supported for now
	enum SET { STRUCTURED_POINTS, STRUCTURED_GRID, UNSTRUCTURED_GRID, POLYDATA, UNSTRUCTURED_POINTS, RECTILINEAR_GRID, FIELD};
	enum DATA { CELL, POINT, NONE };
};

// https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html
class VTK_Formater {
public:

	VTK_Formater(): data_mode(VTK::NONE), nb_cells(0), nb_vertices(0) {}

	VTK_Formater& open(const std::string& name);

	VTK_Formater& save_mesh(Mesh::Geometry& geometry, std::map<Element, Mesh::Topology>& topologies);

	VTK_Formater& start_point_data();

	VTK_Formater& start_cell_data();

	VTK_Formater& add_scalar_data(std::vector<scalar>& data, const std::string& name = "scalars");

	VTK_Formater& add_vector_data(std::vector<Vector3>& data, const std::string& name = "v3_scalars");

	VTK_Formater& close();
	static int get_cell_type(Element e);

protected:
	VTK::DATA data_mode;
	int nb_cells;
	int nb_vertices;
	std::ofstream file;
};