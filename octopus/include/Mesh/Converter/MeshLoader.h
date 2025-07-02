#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include "Mesh/Generator/MeshGenerator.h"
#include <string>
#include <fstream>

class MeshLoader : public MeshGenerator {
public:
    explicit MeshLoader(const std::string &file_path) : _file_path(file_path) {
    }

    Mesh *build() override;

    virtual std::ifstream get_file() {
        return std::ifstream(_file_path.c_str(), std::ios::in);
    }

    virtual void load(std::ifstream &inputFile, Mesh::Geometry &vertices,
                      std::map<Element, Mesh::Topology> &topologies) = 0;

protected:
    std::string _file_path;
};


class Msh_Loader : public MeshLoader {
public:
    explicit Msh_Loader(const std::string &file_path) : MeshLoader(file_path) {
    }

    void load(std::ifstream &inputFile, Mesh::Geometry &vertices,
              std::map<Element, Mesh::Topology> &topologies) override;
};


class VTK_Loader : public MeshLoader {
public:
    explicit VTK_Loader(const std::string &file_path) : MeshLoader(file_path) {
    }

    void load(std::ifstream &inputFile, Mesh::Geometry &vertices,
              std::map<Element, Mesh::Topology> &topologies) override;

    void skip_lines(std::ifstream &inputFile, std::string &line);

    void skip_to(std::ifstream &inputFile, std::string &line, const std::string &s);

    std::vector<Vector3> get_point_data_v3(const std::string &att_name);
    bool check_vector_data(const std::string& att_name);
};
