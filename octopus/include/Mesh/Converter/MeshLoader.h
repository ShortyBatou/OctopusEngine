#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class MeshLoader : public MeshGenerator {
public:
    MeshLoader(const std::string& file_path) : _file_path(file_path) { }
    Mesh* build() {
        Mesh* mesh = new Mesh();

        std::ifstream inputFile(_file_path.c_str(), std::ios::in);
        if (!inputFile.good()) {
            std::cerr << "Error: could not read " << _file_path << std::endl;
            return nullptr;
        }
        
        load(inputFile, mesh->geometry(), mesh->topologies());
        std::cout << "MESH : " << _file_path << std::endl;
        std::cout << "NB VERTICES = " << mesh->geometry().size() << std::endl;
        for (auto topo : mesh->topologies()) {
            int nb = element_vertices(topo.first);
            std::cout << "NB ELEMENT (" << nb << ") = " << topo.second.size() / nb << std::endl;
        }

        apply_transform(mesh->geometry());
        return mesh;
    }

    virtual void load(std::ifstream& inputFile, Mesh::Geometry& vertices, std::map<Element, Mesh::Topology>& topologies) = 0;

protected:
    std::string _file_path;
};


class Msh_Loader : public MeshLoader {
public:
    Msh_Loader(const std::string& file_path) : MeshLoader(file_path) { }
    void load(std::ifstream& inputFile, Mesh::Geometry& vertices, std::map<Element, Mesh::Topology>& topologies)
    {
        static std::map< int, int > elemTypeNumMap;
        elemTypeNumMap[2] = Line;
        elemTypeNumMap[3] = Triangle;
        elemTypeNumMap[4] = Tetra;
        elemTypeNumMap[5] = Pyramid;
        elemTypeNumMap[6] = Prysm;
        elemTypeNumMap[8] = Hexa;
        elemTypeNumMap[10] = Tetra10;
        elemTypeNumMap[20] = Tetra20;

        std::string line;
        unsigned int nbElements = 0, eltType, eltNum, n;
        int eltTag;
        scalar d;

        // skip lines untill vertices are found
        while (std::getline(inputFile, line)) {
            if (line.compare(0, 6, "$Nodes") == 0) break;
        }

        // get vertices from file
        std::getline(inputFile, line); // nb vertices
        unsigned int off = 1;
        while (std::getline(inputFile, line)) {
            if (line.compare(0, 9, "$EndNodes") == 0) break;

            std::istringstream stream(line, std::istringstream::in);
            stream >> n;
            if (n == 0)  // check for first indice since tetgen generation
                off = 0; // sometimes starts indices from 0

            scalar x, y, z;
            stream >> x >> y >> z;
            vertices.push_back({ x,y,z });
        }

        // skip untill vertices are found
        while (std::getline(inputFile, line)) {
            if (line.compare(0, 9, "$Elements") == 0) break;
        }

        // get elements
        std::getline(inputFile, line); // nb elements
        std::istringstream stream(line, std::istringstream::in);
        stream >> nbElements;
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
            for (unsigned int i = 0; i < n; ++i) {
                stream >> eltTag;
            }
            
            for (int j = 0; j < eltType; ++j) {
                stream >> n;
                topologies[Element(elemTypeNumMap[eltType])].push_back(n - off);
            }
            
        }
    }

};