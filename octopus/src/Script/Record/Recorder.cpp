
#include <UI/AppInfo.h>
#include "Script/Record/Recorder.h"
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <Core/Engine.h>
#include <Mesh/Generator/BeamGenerator.h>

void MeshRecorder::add_data_json(std::ofstream &json) {
    json <<
            "{";

    for (const auto &it: _mesh->topologies()) {
        if (it.second.empty()) continue;
        json << "\"" << element_name(it.first) << "\" : " << it.second.size() / elem_nb_vertices(it.first) << ",";
    }
    json << "\"vertices\" : " << _mesh->nb_vertices();

    json << "}";
}


void FEM_Dynamic_Recorder::add_data_json(std::ofstream &json) {
    json <<
            "{"
            << "\"material\" : \"" << get_material_name(fem_dynamic->_material) << "\","
            << "\"poisson\" : " << fem_dynamic->_poisson << ","
            << "\"young\" : " << fem_dynamic->_young << ","
            << "\"sub_step\" : " << fem_dynamic->_sub_iteration << ","
            << "\"density\" : " << fem_dynamic->_density <<
            "}";
}


void XPBD_FEM_Dynamic_Recorder::add_data_json(std::ofstream &json) {
    json <<
            "{"
            << "\"material\" : \"" << get_material_name(fem_dynamic->_material) << "\","
            << "\"poisson\" : " << fem_dynamic->_poisson << ","
            << "\"young\" : " << fem_dynamic->_young << ","
            << "\"iteration\" : " << fem_dynamic->_iteration << ","
            << "\"sub_itetrations\" : " << fem_dynamic->_sub_iteration << ","
            << "\"density\" : " << fem_dynamic->_density <<
            "}";
}


    void Mesh_Diff_VTK_Recorder::init(Entity *entity) {
        _ps_dynamic = entity->get_component<ParticleSystemDynamics_Getters>();
        _mesh = entity->get_component<Mesh>();
        _mesh_diff = Engine::GetEntity(_id_other)->get_component<Mesh>();
        _off = _mesh->vertice(0) - _mesh_diff->vertice(0);
        assert(_mesh_diff && _ps_dynamic && _mesh);
    }

    void Mesh_Diff_VTK_Recorder::save() {
        // get particle saved data
        std::vector<Vector3> displacements = _ps_dynamic->get_displacement();
        const Mesh::Geometry& g_first = _mesh->geometry();
        const Mesh::Geometry& g_second = _mesh_diff->geometry();

        std::vector<scalar> diff(g_first.size());
        for(int i = 0; i < diff.size(); ++i) {
            diff[i] = glm::length(g_first[i] - g_second[i] - _off);
        }

        VTK_Formater vtk;
        vtk.open(_file_name + "_diff_" + std::to_string(Time::Frame()));
        vtk.save_mesh(_ps_dynamic->get_init_positions(), _mesh->topologies());
        vtk.start_point_data();
        vtk.add_scalar_data(diff, "u_norm");
        vtk.add_vector_data(displacements, "u");
        vtk.close();
    }

void Mesh_VTK_Recorder::add_data_json(std::ofstream &json) {
    json << "\"" << AppInfo::PathToAssets() + _file_name + ".vtk\"";
}

void Mesh_VTK_Recorder::save() {
    VTK_Formater vtk;
    vtk.open(_file_name);
    vtk.save_mesh(_mesh->geometry(), _mesh->topologies());
    vtk.close();
}

void FEM_VTK_Recorder::init(Entity *entity) {
    _fem_dynamic = entity->get_component<FEM_Dynamic_Getters>();
    _ps_dynamic = entity->get_component<ParticleSystemDynamics_Getters>();
    _mesh = entity->get_component<Mesh>();
    assert(_fem_dynamic && _ps_dynamic && _mesh);
}

void FEM_VTK_Recorder::add_data_json(std::ofstream &json) {
    json << "\"" << AppInfo::PathToAssets() + _file_name + "_" + std::to_string(Time::Frame()) + ".vtk\"";
}

void FEM_VTK_Recorder::save() {
    // get particle saved data
    std::vector<Vector3> displacements = _ps_dynamic->get_displacement();

    std::vector<Vector3> init_pos = _ps_dynamic->get_init_positions();
    std::vector<scalar> massses = _ps_dynamic->get_masses();

    // get element saved data
    std::map<Element, std::vector<scalar> > e_stress = _fem_dynamic->get_stress();
    std::map<Element, std::vector<scalar> > e_volume = _fem_dynamic->get_volume();
    std::map<Element, std::vector<scalar> > e_volume_diff = _fem_dynamic->get_volume_diff();
    std::vector<scalar> all_stress, all_volume, all_volume_diff;
    for (auto &it: e_stress) {
        Element type = it.first;
        std::vector<scalar> &stress = e_stress[type];
        std::vector<scalar> &volume = e_volume[type];
        std::vector<scalar> &volume_diff = e_volume_diff[type];
        size_t n = all_stress.size();
        all_stress.resize(all_stress.size() + stress.size());
        all_volume.resize(all_volume.size() + volume.size());
        all_volume_diff.resize(all_volume_diff.size() + volume_diff.size());
        for (size_t i = 0; i < stress.size(); ++i) {
            all_stress[n + i] = stress[i];
            all_volume[n + i] = volume[i];
            all_volume_diff[n + i] = volume_diff[i];
        }
    }

    std::vector<scalar> displacements_norm(displacements.size());
    for(int i = 0; i < displacements.size(); ++i) {
        displacements_norm[i] = glm::length(displacements[i]);
    }

    std::vector<scalar> smooth_stress = _fem_dynamic->get_stress_vertices();

    VTK_Formater vtk;
    vtk.open(_file_name + "_" + std::to_string(Time::Frame()));
    vtk.save_mesh(init_pos, _mesh->topologies());
    vtk.start_point_data();
    vtk.add_scalar_data(massses, "weights");
    vtk.add_scalar_data(displacements_norm, "u_norm");
    vtk.add_vector_data(displacements, "u");
    vtk.add_scalar_data(smooth_stress, "smooth_stress");
    vtk.start_cell_data();
    vtk.add_scalar_data(all_stress, "stress");
    vtk.add_scalar_data(all_volume, "volume");
    vtk.add_scalar_data(all_volume_diff, "volume_diff");
    vtk.close();
}


void Graphic_VTK_Recorder::add_data_json(std::ofstream &json) {
    json << "\"" << AppInfo::PathToAssets() + _file_name + "_Graphic_" + std::to_string(Time::Frame()) + ".vtk\"";
}

void Graphic_VTK_Recorder::save() {
    std::map<Element, Mesh::Topology> topologies;
    topologies[Line] = Mesh::Topology();
    for (auto &it: _graphic->gl_topologies()) {
        topologies[Line].insert(topologies[Line].end(), it.second->lines.begin(), it.second->lines.end());
    }
    VTK_Formater vtk;
    vtk.open(_file_name + "_Graphic_" + std::to_string(Time::Frame()));
    vtk.save_mesh(_graphic->gl_geometry()->geometry, topologies);
    vtk.close();
}


void FEM_Flexion_error_recorder::init(Entity *entity) {
    _ps = entity->get_component<ParticleSystemDynamics_Getters>();
    assert(_ps != nullptr);
    const std::vector<Vector3> positions = _ps->get_init_positions();
    bool found = false;
    for (int i = 0; i < positions.size(); ++i) {
        if (glm::length2(positions[i] - _p_follow) >= 1e-6) continue;
        found = true;
        p_id = i;
        break;
    }

    if (!found) {
        std::cout << "ERROR : No particle found for flexion error" << std::endl;
    } else {
        std::cout << "Flexion Error : Follow " << p_id << "th particle" << std::endl;
    }
}


void FEM_Flexion_error_recorder::add_data_json(std::ofstream &json) {
    const Vector3 p = _ps->get_positions()[p_id];
    json <<
            "{"
            << "\"error\" : " << glm::length2(_p_target - p) << ","
            << "\"p\" : [" << p.x << ", " << p.y << ", " << p.z << "],"
            << "\"target\" : [" << _p_target.x << ", " << _p_target.y << ", " << _p_target.z << "]"
            <<
            "}";
}

void FEM_Torsion_error_recorder::init(Entity *entity) { {
        auto fem_dynamic = entity->get_component<XPBD_FEM_Dynamic>();
        if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
    } {
        auto fem_dynamic = entity->get_component<FEM_Dynamic_Generic>();
        if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
    }

    for (int i = 0; i < _ps->nb_particles(); ++i) {
        Particle *p = _ps->get(i);
        if (p->position.y < 1e-6 && p->position.z < 1e-6) p_ids.push_back(i);
        //if (p->position.y < 1e-6 && p->position.z > 0.9999) p_ids.push_back(i);
        //if (p->position.y > 0.9999 && p->position.z < 1e-6) p_ids.push_back(i);
        //if (p->position.y > 0.9999f && p->position.z > 0.9999) p_ids.push_back(i);
    }

    if (p_ids.empty()) {
        std::cout << "ERROR : No particle found for torsion error" << std::endl;
    } else {
        std::cout << "Flexion Error : Follow " << p_ids.size() << " particles" << std::endl;
    }
}


void FEM_Torsion_error_recorder::compute_errors(std::vector<scalar> &dist, std::vector<scalar> &angles) const {
    for (const int p_id: p_ids) {
        const Particle *p = _ps->get(p_id);
        const Vector2 d_init = glm::normalize(Vector2(p->init_position.y, p->init_position.z) - Vector2(0.5, 0.5));
        const Vector2 d_current = glm::normalize(Vector2(p->position.y, p->position.z) - Vector2(0.5, 0.5));

        scalar angle = std::abs(std::atan2(d_current.x * d_init.y - d_current.y * d_init.x,
                                           d_current.x * d_init.x + d_current.y * d_init.y));
        angle = glm::degrees(angle);
        dist.push_back(p->init_position.x / _beam_length);
        angles.push_back(angle);
    }
    for (int i = 0; i < dist.size() - 1; ++i) {
        for (int j = 0; j < dist.size() - 1; ++j) {
            if (dist[j] <= dist[j + 1]) continue;
            std::swap(dist[j], dist[j + 1]);
            std::swap(angles[j], angles[j + 1]);
        }
    }
    std::vector<scalar> temp_angle;
    std::vector<scalar> temp_dist;
    int i = 0;
    while (i < dist.size()) {
        int j = 1;
        while (i + j < dist.size() && abs(dist[i] - dist[i + j]) < 1e-4) {
            angles[i] += angles[i + j];
            ++j;
        }
        temp_angle.push_back(angles[i] / static_cast<scalar>(j));
        temp_dist.push_back(dist[i]);
        i += j;
    }
    dist = temp_dist;
    angles = temp_angle;
}

void FEM_Torsion_error_recorder::add_data_json(std::ofstream &json) {
    std::vector<scalar> dist;
    std::vector<scalar> angles;
    compute_errors(dist, angles);
    json <<
            "{"
            << "\"rot_max\" : " << _max_rotation << ","
            << "\"beam_lenght\" : " << _beam_length << ","
            << "\"distance\" : ";
    add_scalar_array(json, dist);
    json << ",";
    json << "\"angles\" : ";
    add_scalar_array(json, angles);
    json <<
            "}";
}

void FEM_Torsion_error_recorder::add_scalar_array(std::ofstream &json, const std::vector<scalar> &s_array) {
    json << "[";
    for (int i = 0; i < s_array.size(); ++i) {
        json << s_array[i];
        if (i < s_array.size() - 1) json << ",";
    }
    json << "]";
}
