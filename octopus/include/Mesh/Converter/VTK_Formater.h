#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include "UI/AppInfo.h"
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

	VTK_Formater() {}

	VTK_Formater& open(std::string name) {
		file.open(AppInfo::PathToAssets() + "/vtk/" + name + ".vtk");
		file << "# vtk DataFile Version 5.1\n";
		file << "vtk output\n";
		file << "ASCII\n\n";
		data_mode = VTK::DATA::NONE;
		return *this;
	}

	VTK_Formater& save_mesh(Mesh* mesh) {
		file << "DATASET UNSTRUCTURED_GRID\n";
		nb_vertices = mesh->geometry().size();
		// POINTS n dataType
		file << "POINTS " << mesh->geometry().size() << " float\n";
		// p0x p0y p0z
		for (Vector3& v : mesh->geometry()) {
			file << v.x << " " << v.y << " " << v.z << "\n";
		}
		file << "\n";

		nb_cells = 0;
		unsigned int nb_indices = 0;
		for (auto& topo : mesh->topologies()) {
			nb_cells += topo.second.size() / element_vertices(topo.first);
			nb_indices += topo.second.size();
		}

		// CELLS n size
		file << "CELLS " << nb_cells << " " << nb_indices << "\n";
		file << "OFFSETS vtktypeint64\n";

		unsigned int offset = 0;
		for (auto& topo : mesh->topologies()) {
			unsigned int element_size = element_vertices(topo.first);
			for (unsigned int i = 0; i < topo.second.size(); i += element_size) {
				file << offset << " ";
				offset += element_size;
			}
			file << "\n";
		}

		file << "CONNECTIVITY vtktypeint64\n";

		// numPoints0 i0 j0 k0 …
		for (auto& topo : mesh->topologies()) {
			unsigned int element_size = element_vertices(topo.first);
			unsigned int nb_elements = topo.second.size() / element_vertices(topo.first);;
			for (unsigned int i = 0; i < topo.second.size(); i += element_size) {
				for (unsigned int j = 0; j < element_size; ++j)
				{
					file << topo.second[i + j] << " ";
				}
				if (topo.second.size() > 0) file << "\n";
			}
		}

		file << "CELL_TYPES " << nb_cells << "\n";
		for (auto& topo : mesh->topologies()) {
			unsigned int element_size = element_vertices(topo.first);
			unsigned int element_vtk_id = get_cell_type(topo.first);
			unsigned int nb_elements = topo.second.size() / element_vertices(topo.first);;
			for (unsigned int i = 0; i < topo.second.size(); i += element_size) {
				file << element_vtk_id << " ";
			}
			if (topo.second.size() > 0) file << "\n";
		}
		file << "\n";
		return *this;
	}

	VTK_Formater& start_point_data() {
		file << "POINT_DATA " << nb_vertices << "\n";
		data_mode = VTK::DATA::POINT;
		return *this;
	}

	VTK_Formater& start_cell_data() {
		file << "CELL_DATA " << nb_cells << "\n";
		data_mode = VTK::DATA::CELL;
		return *this;
	}

	VTK_Formater& add_scalar_data(std::vector<scalar>& data) {
		if (data_mode == VTK::DATA::POINT) assert(data.size() == nb_vertices);
		else if (data_mode == VTK::DATA::CELL) assert(data.size() == nb_cells);
		file << "SCALARS scalars float 1\n";
		file << "LOOKUP_TABLE default\n";
		for (scalar& d : data) {
			file << d << " ";
		}
		file << "\n";
		return *this;
	}

	VTK_Formater& add_vector_data(std::vector<Vector3>& data) {
		if (data_mode == VTK::DATA::POINT) assert(data.size() == nb_vertices);
		else if (data_mode == VTK::DATA::CELL) assert(data.size() == nb_cells);
		file << "VECTORS vectors float\n";
		for (Vector3& d : data) {
			file << d.x << " " << d.y << " " << d.z << "  ";
		}
		file << "\n";
		return *this;
	}

	VTK_Formater& close() {
		file.close();
		return *this;
	}

	unsigned int get_cell_type(Element e) {
		switch (e)
		{
		case Line: return 3;
		case Triangle:return 5;
		case Quad: return 9;
		case Tetra: return 10;
		case Pyramid: return 14;
		case Prysm: return 13;
		case Hexa: return 12;
		case Tetra10:return 24;
		default:
			return 0;
		}

	}

protected:
	VTK::DATA data_mode;
	unsigned int nb_cells;
	unsigned int nb_vertices;
	std::ofstream file;
};