#include "Mesh/Converter/VTK_Formater.h"
#include "UI/AppInfo.h"

VTK_Formater& VTK_Formater::open(const std::string& name) {
	file.open(AppInfo::PathToAssets() + name + ".vtk");
	file << "# vtk DataFile Version 5.1\n";
	file << "vtk output\n";
	file << "ASCII\n\n";
	data_mode = VTK::DATA::NONE;
	return *this;
}

VTK_Formater& VTK_Formater::save_mesh(Mesh::Geometry& geometry, std::map<Element, Mesh::Topology>& topologies) {
	file << "DATASET UNSTRUCTURED_GRID\n";
	nb_vertices = static_cast<int>(geometry.size());
	// POINTS n dataType
	file << "POINTS " << geometry.size() << " float\n";
	// p0x p0y p0z
	for (Vector3& v : geometry) {
		file << v.x << " " << v.y << " " << v.z << "\n";
	}
	file << "\n";

	nb_cells = 0;
	int nb_indices = 0;
	for (auto& topo : topologies) {
		nb_cells += static_cast<int>(topo.second.size()) / elem_nb_vertices(topo.first);
		nb_indices += static_cast<int>(topo.second.size());
	}

	// CELLS n size
	file << "CELLS " << nb_cells + 1 << " " << nb_indices << "\n";
	file << "OFFSETS vtktypeint64\n";

	int offset = 0;
	for (auto& topo : topologies) {
		if (static_cast<int>(topo.second.size()) == 0) continue;
		int element_size = elem_nb_vertices(topo.first);
		for (int i = 0; i <= topo.second.size(); i += element_size) {
			file << offset << " ";
			offset += element_size;
		}
		file << "\n";
	}

	file << "CONNECTIVITY vtktypeint64\n";

	// numPoints0 i0 j0 k0 �
	for (auto& topo : topologies) {
		int element_size = elem_nb_vertices(topo.first);
		for (int i = 0; i < topo.second.size(); i += element_size) {
			for (int j = 0; j < element_size; ++j)
			{
				file << topo.second[i + j] << " ";
			}
			if (!topo.second.empty()) file << "\n";
		}
	}

	file << "CELL_TYPES " << nb_cells << "\n";
	for (auto& topo : topologies) {
		const int element_size = elem_nb_vertices(topo.first);
		const int element_vtk_id = get_cell_type(topo.first);
		for (int i = 0; i < topo.second.size(); i += element_size) {
			file << element_vtk_id << " ";
		}
		if (!topo.second.empty()) file << "\n";
	}
	file << "\n";
	return *this;
}

VTK_Formater& VTK_Formater::start_point_data() {
	file << "POINT_DATA " << nb_vertices << "\n";
	data_mode = VTK::DATA::POINT;
	return *this;
}

VTK_Formater& VTK_Formater::start_cell_data() {
	file << "CELL_DATA " << nb_cells << "\n";
	data_mode = VTK::DATA::CELL;
	return *this;
}

VTK_Formater& VTK_Formater::add_scalar_data(std::vector<scalar>& data, const std::string& name) {
	if (data_mode == VTK::DATA::POINT) assert(data.size() == nb_vertices);
	else if (data_mode == VTK::DATA::CELL) assert(data.size() == nb_cells);
	file << "SCALARS " << name << " float 1\n";
	file << "LOOKUP_TABLE default\n";
	for (scalar& d : data) {
		file << d << " ";
	}
	file << "\n";
	return *this;
}

VTK_Formater& VTK_Formater::add_vector_data(std::vector<Vector3>& data, const std::string& name) {
	if (data_mode == VTK::DATA::POINT) assert(data.size() == nb_vertices);
	else if (data_mode == VTK::DATA::CELL) assert(data.size() == nb_cells);
	file << "VECTORS " << name << " float\n";
	for (Vector3& d : data) {
		file << d.x << " " << d.y << " " << d.z << "  ";
	}
	file << "\n";
	return *this;
}

VTK_Formater& VTK_Formater::close() {
	file.close();
	return *this;
}

int VTK_Formater::get_cell_type(Element e) {
	switch (e)
	{
	case Line: return 3;
	case Triangle:return 5;
	case Quad: return 9;
	case Tetra: return 10;
	case Pyramid: return 14;
	case Prism: return 13;
	case Hexa: return 12;
	case Tetra10:return 24;
	case Tetra20:return 71;
	default:
		return 0;
	}

}

