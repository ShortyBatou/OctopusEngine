#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include "Tools/Area.h"
#include <string>
#include <vector>
#include <iostream>

struct Error {
    Error(): MSE(0), RMSE(0), MAE(0), max(0), L1(0), L2(0), n(0) {
    }
    void update(const Vector3& e) {
        n++;
        max = std::max(max, glm::length(e));
        L2 += glm::length2(e);
        L1 += glm::length(e);
        MAE = L1 / n;
        MSE = L2 / n;
        RMSE = sqrt(MSE);
    }
    scalar MSE, RMSE, MAE, max, L1, L2;
    int n;
};

struct MeshDiff : Behaviour {
    MeshDiff(const int ref, const std::vector<int> &ids) : _ref_id(ref), _ids(ids) { }

    virtual void build() = 0;
    virtual Error get_error(int mesh_id) = 0;

    void late_init() override {
        Mesh* r_mesh = Engine::GetEntity(_ref_id)->get_component<Mesh>();
        _meshes[_ref_id] = r_mesh;
        for(int id : _ids) {
            Entity* e = Engine::GetEntity(id);
            Mesh* mesh = e->get_component<Mesh>();
            _meshes[id] = mesh;
            init_offset[id] = r_mesh->geometry()[0] - mesh->geometry()[0];
        }
        build();
    }

    void late_update() override {
        for(const int id : _ids) {
            const Error error = get_error(id);
            DebugUI::Begin( std::to_string(_ref_id) + " => " + std::to_string(id) + ": error");
            DebugUI::Value(std::to_string(id) + " MSE = ", error.MSE);
            DebugUI::Value(std::to_string(id) + " RMSE = ", error.RMSE);
            DebugUI::Value(std::to_string(id) + " MAE = ", error.MAE);
            DebugUI::Value(std::to_string(id) + " max = ", error.max);
            DebugUI::End();
        }
    }

protected:
    std::vector<int> _ids;
    int _ref_id;
    std::map<int, Mesh*> _meshes;
    std::map<int,Vector3> init_offset;
};



struct MeshDiff_MSE final : public MeshDiff {
    MeshDiff_MSE(const int ref, const std::vector<int> &ids) : MeshDiff(ref,ids) {}

    void build() override {
        Mesh* r_mesh = _meshes[_ref_id];
        for(auto [id, mesh] : _meshes) {
            assert(mesh->geometry().size() == r_mesh->geometry().size());
        }
    }

    Error get_error(const int mesh_id) override {
        Error error;
        Mesh* r_mesh = _meshes[_ref_id];
        Mesh* mesh = _meshes[mesh_id];
        for(int i = 0; i < r_mesh->geometry().size(); ++i) {
            error.update(r_mesh->geometry()[i] - mesh->geometry()[i] - init_offset[mesh_id]);
        }
        return error;
    }
};

struct Beam_MSE_Sampling final : public MeshDiff {
    Beam_MSE_Sampling(const int ref, const std::vector<int> &ids, int nb_sample) : MeshDiff(ref, ids), _nb_sample(nb_sample) {

    }

    void build() override {
        // construire la liste des sommets dans la poutres
        Box box(_meshes[_ref_id]->geometry());

        Vector3 size = box.pmax - box.pmin - Vector3(2.f * large_eps);
        Vector3 step = size * (1.0f / _nb_sample);
        for(int x = 0; x <= _nb_sample * size.x; ++x)
        for(int y = 0; y <= _nb_sample * size.y; ++y)
        for(int z = 0; z <= _nb_sample * size.z; ++z) {
            global_sample.push_back(Vector3(step.x * x / size.x, step.y * y  / size.y, step.z * z / size.z) - box.pmin + Vector3(large_eps));
        }

        /*for(int i = 0; i < _nb_sample; ++i) {
            Vector3 s = Random::InBox(box.pmin, box.pmax) - box.pmin;
            global_sample.push_back(s);
        }*/

        for(auto [id, mesh] : _meshes) {
            Element element = Tetra;
            for(auto& [e, topology] : mesh->topologies()) {
                if(topology.size() == 0) continue;
                element = e;
                _shapes[id] = get_fem_shape(element);
                _types[id] = e;
                break;
            }

            std::vector<Vector3> r_pos = _shapes[id]->get_vertices();
            Mesh::Topology& topo = mesh->topology(element);
            Mesh::Geometry& geo = mesh->geometry();
            const int elem_nb_vert = _shapes[id]->nb;
            const int nb_elem = topo.size() / elem_nb_vert;
            Box o_box(mesh->geometry());
            _offsets[id] = o_box.pmin;
            const Element lin_element = get_linear_element(element);
            const int lin_element_nb_vert = elem_nb_vertices(lin_element);

            // build acceleration grid
            Vector3I grid_size(10,10,10);
            std::vector<std::vector<int>> grid(grid_size.x * grid_size.y * grid_size.z);
            for(int e = 0; e < nb_elem; ++e) {
                int off = e * elem_nb_vert;
                Box e_box(geo.data(), topo.data() + off, lin_element_nb_vert);
                e_box.pmin -= _offsets[id]; e_box.pmax -= _offsets[id];
                e_box.pmin = (e_box.pmin - box.pmin) / (box.pmax - box.pmin);
                e_box.pmax = (e_box.pmax - box.pmin) / (box.pmax - box.pmin);

                Vector3 i_pmin = glm::floor(
                    Vector3(e_box.pmin.x * grid_size.x, e_box.pmin.y * grid_size.y, e_box.pmin.z * grid_size.z));
                Vector3 i_pmax = glm::floor(
                    Vector3(e_box.pmax.x * grid_size.x, e_box.pmax.y * grid_size.y, e_box.pmax.z * grid_size.z));
                i_pmax = min(i_pmax, Vector3(9,9,9));
                for(int x = i_pmin.x; x <= i_pmax.x; ++x) {
                for(int y = i_pmin.y; y <= i_pmax.y; ++y) {
                for(int z = i_pmin.z; z <= i_pmax.z; ++z) {
                    grid[x + y * grid_size.x + z * grid_size.x * grid_size.y].push_back(e);
                }}}
            }

            for(const Vector3& s : global_sample) {
                // get position of s in grid
                Vector3 p = (s - box.pmin) / (box.pmax - box.pmin);
                Vector3 i_p = glm::floor(Vector3(p.x * grid_size.x, p.y * grid_size.y, p.z * grid_size.z));
                i_p = min(i_p, Vector3(9,9,9));
                // for all element in this cell check if point in
                for(int e : grid[i_p.x + i_p.y * grid_size.x + i_p.z * grid_size.x * grid_size.y]) {
                    int off = e * elem_nb_vert;
                    if(lin_element == Hexa) {
                        // get pmin and pmax and check if in box
                        Box e_box(geo.data(), topo.data() + off, lin_element_nb_vert);
                        e_box.pmin -= _offsets[id]; e_box.pmax -= _offsets[id];
                        if(!e_box.inside(s)) continue;
                        // get coordinate in reference element
                        Vector3 a = geo[topo[off]] - _offsets[id], b = geo[topo[off + 6]] - _offsets[id];
                        // in unit cube
                        Vector3 sample = (s - a) / (b - a);
                        // in ref element
                        std::swap(sample.y, sample.z);
                        sample *= 2; sample += r_pos[0];
                        _samples[id].push_back(sample);
                        _elements[id].push_back(e);
                        break;
                    }
                    else {
                        Tetraedron e_tetra(geo.data(), topo.data() + off);
                        if(!e_tetra.inside(s + _offsets[id])) continue;
                        Vector4 coord = e_tetra.barycentric(s + _offsets[id]);
                        Vector3 sample(0);
                        for(int i = 0; i < 4; ++i) {
                            sample += r_pos[i] * coord[i];
                        }
                        _samples[id].push_back(sample);
                        _elements[id].push_back(e);
                        break;
                    }
                }

            }
        }
    }

    Error get_error(const int id) override {
        ColorMap::Set_Type(ColorMap::Rainbow);
        Mesh* mesh = _meshes[id];
        const FEM_Shape* shape = _shapes[id];
        const Mesh::Topology& topo = mesh->topology(_types[id]);
        const Mesh::Geometry& geo = mesh->geometry();

        Mesh* r_mesh = _meshes[_ref_id];
        const FEM_Shape* r_shape = _shapes[_ref_id];
        const Mesh::Topology& r_topo = r_mesh->topology(_types[_ref_id]);
        const Mesh::Geometry& r_geo = r_mesh->geometry();

        Error error;
        for(int i = 0; i < global_sample.size(); ++i) {
            const Vector3 s = _samples[id][i];

            std::vector<scalar> weights = shape->build_shape(s.x, s.y, s.z);
            const int off = _elements[id][i] * shape->nb;
            Vector3 p(0.f);
            for(int j = 0; j < weights.size(); ++j) {
                p += geo[topo[off + j]] * weights[j];
            }

            const Vector3 r_s = _samples[_ref_id][i];
            std::vector<scalar> r_weights = r_shape->build_shape(r_s.x, r_s.y, r_s.z);
            const int r_off = _elements[_ref_id][i] * r_shape->nb;
            Vector3 r_p(0.f);
            for(int j = 0; j < r_weights.size(); ++j) {
                r_p += r_geo[r_topo[r_off + j]] * r_weights[j];
            }

            error.update(p - r_p - _offsets[id]);
        }
        return error;
    }

protected:

    std::vector<Vector3> global_sample;
    int _nb_sample;
    std::map<int, Vector3> _offsets;
    std::map<int, FEM_Shape*> _shapes;
    std::map<int, Element> _types;
    std::map<int, std::vector<Vector3>> _samples;
    std::map<int, std::vector<int>> _elements;
};