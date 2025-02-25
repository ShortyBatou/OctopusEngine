#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include <string>
#include <vector>
#include <iostream>

struct MeshDiff : public Behaviour {
    MeshDiff(int ref, std::vector<int> ids) : _ref_id(ref), _ids(ids) {}

    void late_init() override {
        Mesh* r_mesh = Engine::GetEntity(_ref_id)->get_component<Mesh>();
        _meshes[_ref_id] = r_mesh;
        scalar max_x = 0;
        for(Vector3 p : r_mesh->geometry()) {
            max_x = std::max(max_x, p.x);
        }
        for(int i = 0; i < r_mesh->geometry().size(); ++i) {
            Vector3 p = r_mesh->geometry()[i];
            if(std::abs(p.x - max_x) < eps * 10.f) {
                far_id.push_back(i);
            }
        }

        for(int id : _ids) {
            Entity* e = Engine::GetEntity(id);
            Mesh* mesh = e->get_component<Mesh>();
            _meshes[id] = mesh;
            assert(mesh->geometry().size() == r_mesh->geometry().size());
            init_offset[id] = r_mesh->geometry()[0] - mesh->geometry()[0];
        }
    }

    void late_update() override {
        Mesh* r_mesh = _meshes[_ref_id];
        DebugUI::Begin( "error");
        for(int id : _ids) {
            Mesh* mesh = _meshes[id];
            scalar diff = 0;
            //for(int i : far_id) {
            for(int i = 0; i < r_mesh->geometry().size(); ++i) {
                diff += glm::length2(r_mesh->geometry()[i] - mesh->geometry()[i] - init_offset[id]);
            }
            DebugUI::Value(std::to_string(id) + " = ", diff);

        }
        DebugUI::End();
    }

protected:
    int _ref_id;
    std::map<int,Vector3> init_offset;
    std::vector<int> _ids;
    std::map<int, Mesh*> _meshes;
    std::vector<int> far_id;
};