#pragma once
#include <Rendering/GL_Graphic.h>
#include "Manager/TimeManager.h"
#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"
#include "Mesh/Converter/VTK_Formater.h"
#include <string>
#include <vector>
#include <iostream>



class Recorder {
public:
    virtual ~Recorder() = default;

    virtual void init(Entity *entity) {
    };

    virtual void print() = 0; // printed in std::string for debug
    virtual std::string get_name() = 0; // get recored name or id
    virtual void add_data_json(std::ofstream &json) = 0; // get sub json data
    virtual void save() {
    }
};

class TimeRecorder final : public Recorder {
public:
    void print() override {
        std::cout << "[t = " << Time::Timer() * 1000 << "ms, ft = " << Time::Fixed_Timer() * 1000 << "ms]";
    }

    std::string get_name() override {
        return "Time";
    }

    void add_data_json(std::ofstream &json) override {
        json <<
                "{"
                << "\"frame\" : " << Time::Frame() << ","
                << "\"time\" : " << Time::Timer() << ","
                << "\"dt\" : " << Time::DeltaTime() << ","
                << "\"fixed_Time\" : " << Time::Fixed_Timer() << ","
                << "\"fixed_dt\" : " << Time::Fixed_DeltaTime() << ""
                <<
                "}";
    }
};

class MeshRecorder final : public Recorder {
public:
    MeshRecorder() : _mesh(nullptr) {
    }

    void init(Entity *entity) override {
        _mesh = entity->get_component<Mesh>();
    }

    void print() override {
    }

    std::string get_name() override { return "mesh"; }

    void add_data_json(std::ofstream &json) override;

private:
    Mesh *_mesh;
};

class FEM_Dynamic_Recorder final : public Recorder {
public:
    FEM_Dynamic_Recorder() : fem_dynamic(nullptr) {
    }

    void init(Entity *entity) override {
        fem_dynamic = entity->get_component<FEM_Dynamic_Generic>();
    }

    void print() override {
    }

    std::string get_name() override {
        return "XPBD_FEM_Dynamic";
    }

    void add_data_json(std::ofstream &json) override;

private:
    FEM_Dynamic_Generic *fem_dynamic;
};


class XPBD_FEM_Dynamic_Recorder : public Recorder {
public:
    XPBD_FEM_Dynamic_Recorder() : fem_dynamic(nullptr) {
    }

    void init(Entity *entity) override {
        fem_dynamic = entity->get_component<XPBD_FEM_Dynamic>();
    }

    void print() override {
    }

    std::string get_name() override {
        return "XPBD_FEM_Dynamic";
    }

    void add_data_json(std::ofstream &json) override;

private:
    XPBD_FEM_Dynamic *fem_dynamic;
};

class Mesh_VTK_Recorder final : public Recorder {
public:
    explicit Mesh_VTK_Recorder(const std::string &file_name) : _file_name(file_name), _mesh(nullptr) {
    }

    void init(Entity *entity) override {
        _mesh = entity->get_component<Mesh>();
    }

    void print() override {
    }

    std::string get_name() override {
        return "vtk_file";
    }

    void add_data_json(std::ofstream &json) override;

    void save() override;

private:
    std::string _file_name;
    Mesh *_mesh;
};

class Mesh_Diff_VTK_Recorder final : public Recorder {
public:
    explicit Mesh_Diff_VTK_Recorder(std::string file_name, const int id_other)
       : _file_name(std::move(file_name)),
         _id_other(id_other),
         _mesh(nullptr), _mesh_diff(nullptr)
    { }

    void init(Entity *entity) override;

    void print() override {
    }

    std::string get_name() override {
        return "diff_vtk_file";
    }

    void add_data_json(std::ofstream &json) override { }

    void save() override;

protected:
    std::string _file_name;
    int _id_other;
    ParticleSystemDynamics_Getters* _ps_dynamic;
    Vector3 _off;
    Mesh * _mesh;
    Mesh * _mesh_diff;
};


class Mesh_Sample_VTK_Recorder final : public Recorder {
public:
    explicit Mesh_Sample_VTK_Recorder(std::string file_name)
       : _file_name(std::move(file_name))
    { }

    void init(Entity *entity) override;

    void print() override { }

    std::string get_name() override {
        return "mse_diff_vtk_file";
    }

    void add_data_json(std::ofstream &json) override { }

    void save() override;

protected:
    int id;
    std::string _file_name;
};


class FEM_VTK_Recorder final : public Recorder {
public:
    explicit FEM_VTK_Recorder(std::string file_name)
       : _file_name(std::move(file_name)), _fem_dynamic(nullptr), _ps_dynamic(nullptr), _mesh(nullptr)
    { }

    void init(Entity *entity) override;

    void print() override {
    }

    std::string get_name() override {
        return "vtk_file";
    }

    void add_data_json(std::ofstream &json) override;

    void save() override;

protected:
    std::string _file_name;
    FEM_Dynamic_Getters* _fem_dynamic;
    ParticleSystemDynamics_Getters* _ps_dynamic;

    Mesh *_mesh;
};

class Graphic_VTK_Recorder final : public Recorder {
public:
    explicit Graphic_VTK_Recorder(std::string file_name) : _file_name(std::move(file_name)), _graphic(nullptr) {
    }

    void init(Entity *entity) override {
        _graphic = entity->get_component<GL_Graphic>();
    }

    void print() override {
    }

    std::string get_name() override {
        return "vtk_graphic";
    }

    void add_data_json(std::ofstream &json) override;

    void save() override;

private:
    std::string _file_name;
    GL_Graphic *_graphic;
};


class FEM_Flexion_error_recorder final : public Recorder {
public:
    FEM_Flexion_error_recorder(const Vector3& p_follow, const Vector3& p_target) : _p_follow(p_follow), _p_target(p_target), p_id(0),
                                                                                 _ps(nullptr) {
    }

    void init(Entity *entity) override;

    void print() override {
        const Vector3 p = _ps->get_positions()[p_id];
        std::cout << "[f_err = " << glm::length2(_p_target - p) << "]";
    }

    std::string get_name() override {
        return "flexion_error";
    }

    void add_data_json(std::ofstream &json) override;

private:
    int p_id;
    Vector3 _p_follow, _p_target;
    ParticleSystemDynamics_Getters *_ps;
};


class FEM_Torsion_error_recorder : public Recorder {
public:
    FEM_Torsion_error_recorder(const scalar max_rotation, const scalar beam_length, const int sample_per_edge = 2)
        : _ps(nullptr), _max_rotation(max_rotation), _beam_length(beam_length), _sample_per_edge(sample_per_edge) {
    }

    void init(Entity *entity) override;

    void print() override {
        std::cout << "[f_err = " << "" << "]";
    }

    std::string get_name() override {
        return "rotation_error";
    }

    void compute_errors(std::vector<scalar> &dist, std::vector<scalar> &angles) const;

    void add_data_json(std::ofstream &json) override;

    void add_scalar_array(std::ofstream &json, const std::vector<scalar> &s_array);

private:
    std::vector<int> e_off;
    std::vector<Vector3> r_pos;

    Element elem;
    ParticleSystemDynamics_Getters *_ps;
    Mesh* mesh;
    FEM_Shape* shape;

    scalar _max_rotation, _beam_length;
    int _sample_per_edge;
};
