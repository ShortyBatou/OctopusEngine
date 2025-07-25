#include "Mesh/Converter/MeshLoader.h"
#include <sstream>
#include <iostream>

Mesh* MeshLoader::build() {
    Mesh* mesh = new Mesh();

    std::ifstream inputFile = get_file();
    if (!inputFile.good()) {
        std::cerr << "Error: could not read " << _file_path << std::endl;
        return nullptr;
    }

    load(inputFile, mesh->geometry(), mesh->topologies());
    inputFile.close();
    std::cout << "MESH : " << _file_path << std::endl;
    std::cout << "NB VERTICES = " << mesh->geometry().size() << std::endl;
    for (const auto& topo : mesh->topologies()) {
        const int nb = elem_nb_vertices(topo.first);
        std::cout << "NB ELEMENT (" << nb << ") = " << topo.second.size() / nb << std::endl;
    }

    apply_transform(mesh->geometry());
    return mesh;
}

void Msh_Loader::load(std::ifstream& inputFile, Mesh::Geometry& vertices, std::map<Element, Mesh::Topology>& topologies)
{
    static std::map< int, Element > elemTypeNumMap;
    elemTypeNumMap[2] = Line;
    elemTypeNumMap[3] = Triangle;
    elemTypeNumMap[4] = Tetra;
    elemTypeNumMap[5] = Pyramid;
    elemTypeNumMap[6] = Prism;
    elemTypeNumMap[8] = Hexa;
    elemTypeNumMap[11] = Tetra10;
    elemTypeNumMap[29] = Tetra20;

    std::string line;
    int nbElements = 0, eltNum, n;
    int eltTag, eltType;
    // skip lines untill vertices are found
    while (std::getline(inputFile, line)) {
        if (line.compare(0, 6, "$Nodes") == 0) break;
    }

    // get vertices from file
    std::getline(inputFile, line); // nb vertices
    int off = 1;
    while (std::getline(inputFile, line)) {
        if (line.compare(0, 9, "$EndNodes") == 0) break;

        std::istringstream stream(line, std::istringstream::in);
        stream >> n;
        if (n == 0)  // check for first indice since tetgen generation
            off = 0; // sometimes starts indices from 0

        scalar x, y, z;
        stream >> x >> y >> z;
        vertices.emplace_back( x,y,z );
    }

    // skip untill vertices are found
    while (std::getline(inputFile, line)) {
        if (line.compare(0, 9, "$Elements") == 0) break;
    }

    // get elements
    {
        std::getline(inputFile, line); // nb elements
        std::istringstream stream(line, std::istringstream::in);
        stream >> nbElements;
    }
    while (std::getline(inputFile, line)) {
        if (line.compare(0, 12, "$EndElements") == 0) break;
        eltTag = -1;
        std::istringstream stream(line, std::istringstream::in);
        // element number
        stream >> eltNum;
        // element type
        stream >> eltType;
        // element number of tags
        stream >> n;
        for (int i = 0; i < n; ++i) {
            stream >> eltTag;
        }
        Element elem = elemTypeNumMap[eltType];
        int nb_vertices = elem_nb_vertices(elem);
        for (int j = 0; j < nb_vertices; ++j) {
            stream >> n;
            topologies[elem].push_back(n - off);
        }

    }
}

void VTK_Loader::load(std::ifstream& inputFile, Mesh::Geometry& vertices, std::map<Element, Mesh::Topology>& topologies)
{
    vertices.clear();
    topologies.clear();
    std::string line;
    int nbElements = 0, nb_ids;

    // read Title (256 characters maximum, terminated with newline \n character)
    // if 2D, must be specified at beginning of title!
    skip_lines(inputFile, line);
    skip_lines(inputFile, line);
    skip_lines(inputFile, line);

    if (line.compare(0, 5, "ASCII") != 0) {
        std::cerr << "Reading of BINARY VTK files not implemented yet. ABORT!" << std::endl;
        return;
    }

    skip_lines(inputFile, line);
    if (line.compare(0, 7, "DATASET") != 0) {
        std::cerr << "Missing DATASET in VTK file. ABORT!" << std::endl;
        return;
    }


    std::istringstream stream(line, std::istringstream::in);
    stream.seekg(7, std::ios::cur);
    std::string dataset;
    stream >> dataset;

    // Type is one of: POLYDATA, (and maybe later UNSTRUCTURED_GRID)
    // others types not handled
    if (dataset != "UNSTRUCTURED_GRID") {
        std::cerr << "Only UNSTRUCTURED_GRID format are supported. ABORT!" << std::endl;
        return;
    }

    std::cout << "READ MESH : " << this->_file_path << std::endl;
    skip_lines(inputFile, line);

    if (line.compare(0, 6, "POINTS") == 0) {
        int nbPositions = 0;
        std::istringstream stream_point(line, std::istringstream::in);
        stream_point.seekg(6, std::ios::cur);
        stream_point >> nbPositions;

        vertices.reserve(nbPositions); // pre-allocating

        scalar x, y, z;
        // always 3 coords, last is 0.0 when mMeshInfo.mMeshDim=2
        for (int i = 0; i < nbPositions; ++i) {
            inputFile >> x >> y >> z;
            vertices.emplace_back(x,y,z);
        }
    }

    std::cout << "NB VERTICES = " << vertices.size() << std::endl;


    skip_to(inputFile, line, "CELLS");
    if (line.compare(0, 5, "CELLS") != 0) {
        std::cerr << "No CELLS flag in UNSTRUCTURED_GRID format. ABORT!" << std::endl;
        return;
    }

    // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    static std::map< int, Element > elemTypeFromVTKMap;
    elemTypeFromVTKMap[3] = Line; //VTK_LINE;
    elemTypeFromVTKMap[5] = Triangle; //VTK_TRI
    elemTypeFromVTKMap[9] = Quad; //VTK_QUAD
    elemTypeFromVTKMap[10] = Tetra; //VTK_TETRA
    elemTypeFromVTKMap[12] = Hexa; //VTK_HEXA
    elemTypeFromVTKMap[13] = Prism; //VTK_WEDGE
    elemTypeFromVTKMap[14] = Pyramid; //VTK_PYRAMID
    elemTypeFromVTKMap[24] = Tetra10; //VTK_QUADRATIC_TETRA
    elemTypeFromVTKMap[29] = Hexa27; // VTK_QUADRATIC_HEXA
    elemTypeFromVTKMap[71] = Tetra20; //VTK_LAGRANGE_TETRAHEDRON (technically any order but it's not supported beyond T20)

    std::istringstream stream_elem(line, std::istringstream::in);
    stream_elem.seekg(5, std::ios::cur);
    stream_elem >> nbElements >> nb_ids;
    std::cout << "NB ELEMENTS = " << nbElements << std::endl;

    skip_to(inputFile, line, "CONNECTIVITY vtktypeint64");

    std::vector <int> ids;
    int id;
    //elements.reserve(nbElements);
    for (int j = 0; j < nb_ids; ++j) {
        inputFile >> id;
        ids.push_back(id);
    } // nbElements
    skip_to(inputFile, line, "CELL_TYPES");

    if (line.compare(0, 10, "CELL_TYPES") != 0) {
        std::cerr << "No CELL_TYPES flag in UNSTRUCTURED_GRID format. ABORT!" << std::endl;
        return;
    }

    std::istringstream stream_type(line, std::istringstream::in);
    stream_type.seekg(10, std::ios::cur);
    stream_type >> nbElements;

    int cellType;
    int count = 0;
    for (int j = 0; j < nbElements; ++j) {
        inputFile >> cellType;
        Element elem = elemTypeFromVTKMap[cellType];
        // Moving element info to connectivity
        for (int i = 0; i < elem_nb_vertices(elem); ++i) {
            int vid;
            if(elem == Hexa27 && i == 20) {
                vid = ids[24 + count];
            }
            else if(elem == Hexa27 && i == 21) {
                vid = ids[22 + count];
            }
            else if(elem == Hexa27 && i == 22) {
                vid = ids[21 + count];
            }
            else if(elem == Hexa27 && i == 24) {
                vid = ids[20 + count];
            }
            else if(elem == Hexa27 && i == 16) {
                vid = ids[12 + count];
            }
            else if(elem == Hexa27 && i == 12) {
                vid = ids[16 + count];
            }
            else if(elem == Hexa27 && i == 17) {
                vid = ids[13 + count];
            }
            else if(elem == Hexa27 && i == 13) {
                vid = ids[17 + count];
            }
            else if(elem == Hexa27 && i == 18) {
                vid = ids[14 + count];
            }
            else if(elem == Hexa27 && i == 14) {
                vid = ids[18 + count];
            }
            else if(elem == Hexa27 && i == 19) {
                vid = ids[15 + count];
            }
            else if(elem == Hexa27 && i == 15) {
                 vid = ids[19 + count];
            }
            else {
                vid = ids[i + count];
            }

            topologies[elem].push_back(vid);

        }
        count += elem_nb_vertices(elem);
    }

    std::cout << "MESH LOADED : " << this->_file_path << std::endl;

}

void VTK_Loader::skip_lines(std::ifstream& inputFile, std::string& line) {
    do { //skip end of last line
        if (!std::getline(inputFile, line)) {
            std::cout << "VTKLoader : getLine failed \""<< line<< "\"" << std::endl;
            break;
        }
    } while (line.empty());
}

void VTK_Loader::skip_to(std::ifstream& inputFile, std::string& line, const std::string& s) {
    do { //skip end of last line
        if (!std::getline(inputFile, line)) {
            std::cout << "VTKLoader : getLine failed \"" << line << "\"" << std::endl;
            break;
        }
    } while (line.compare(0, s.size(), s) != 0);
}

bool VTK_Loader::check_vector_data(const std::string& att_name) {
    std::ifstream file = get_file();
    if (!file.good()) {
        std::cerr << "Error: could not read " << _file_path << std::endl;
        return false;
    }

    std::string line;
    // get data length
    skip_to(file, line, "VECTORS " + att_name + " float"); // skip to data
    if (line.compare(0, 14 + att_name.size(), "VECTORS " + att_name + " float") != 0) {
        std::cerr << "Attribute " << att_name << " not found" << std::endl;
        return false;
    }
    return true;
}

std::vector<Vector3> VTK_Loader::get_point_data_v3(const std::string& att_name) {
    std::vector<Vector3> v3;
    std::ifstream file = get_file();
    if (!file.good()) {
        std::cerr << "Error: could not read " << _file_path << std::endl;
        return v3;
    }

    std::string line;
    skip_to(file, line, "POINT_DATA "); // skip to data
    if (line.compare(0, 10, "POINT_DATA") != 0) {
        std::cerr << "POINT_DATA not found" << std::endl;
        return v3;
    }

    std::istringstream stream_nb(line, std::istringstream::in);
    stream_nb.seekg(11, std::ios::cur);
    int nb_point_data;
    stream_nb >> nb_point_data;
    v3.resize(nb_point_data);

    // get data length
    skip_to(file, line, "VECTORS " + att_name + " float"); // skip to data
    if (line.compare(0, 14 + att_name.size(), "VECTORS " + att_name + " float") != 0) {
        std::cerr << "Attribute " << att_name << " not found" << std::endl;
        return v3;
    }
    skip_lines(file, line); // data are on the next line
    std::istringstream steam_data(line, std::istringstream::in);
    scalar x=0, y=0, z=0;
    for (int i = 0; i < nb_point_data; ++i) {
        steam_data >> x >> y >> z;
        v3[i] = Vector3(x, y, z);
    }
    std::cout << "Attribute : " << att_name << "  size = " << nb_point_data << std::endl;
    return v3;
}

