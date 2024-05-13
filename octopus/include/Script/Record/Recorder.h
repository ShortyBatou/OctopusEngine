#pragma once

#include "Manager/TimeManager.h"
#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"
#include "Script/Record/DataRecorder.h"
#include "Mesh/Converter/VTK_Formater.h"

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <glm/gtx/norm.hpp>
class Recorder {
public:
	virtual void init(Entity* entity) {};
	virtual void print() = 0;  // printed in std::string for debug
	virtual std::string get_name() = 0; // get recored name or id
	virtual void add_data_json(std::ofstream& json) = 0; // get sub json data
	virtual void save() {}
};

class TimeRecorder : public Recorder {
public:
	virtual void print() override {
		std::cout << "[t = " << Time::Timer() * 1000 << "ms, ft = " << Time::Fixed_Timer() * 1000 << "ms]";
	}

	virtual std::string get_name() override {
		return "Time";
	}

	virtual void add_data_json(std::ofstream& json) override {
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

class MeshRecorder : public Recorder {
public:

	MeshRecorder() : _mesh(nullptr) { }

	virtual void init(Entity* entity) override {
		_mesh = entity->get_component<Mesh>();
	}

	virtual void print() override { }

	virtual std::string get_name() override {
		return "mesh";
	}

	virtual void add_data_json(std::ofstream& json) override {
		json <<
			"{";

		for (auto it : _mesh->topologies()) {
			if (it.second.size() == 0) continue;
			json <<"\"" << element_name(it.first) << "\" : " << it.second.size() / elem_nb_vertices(it.first) << ",";
		}
		json << "\"vertices\" : " << _mesh->nb_vertices();

		json <<"}";
	}

private:
	Mesh* _mesh;
};

class FEM_Dynamic_Recorder : public Recorder {
public:
	virtual void init(Entity* entity) override {
		fem_dynamic = entity->get_component<FEM_Dynamic>();

	}

	virtual void print() override {
	}

	virtual std::string get_name() override {
		return "XPBD_FEM_Dynamic";
	}

	virtual void add_data_json(std::ofstream& json) override {
		json <<
			"{"
			<< "\"material\" : \"" << get_material_name(fem_dynamic->_material) << "\","
			<< "\"poisson\" : " << fem_dynamic->_poisson << ","
			<< "\"young\" : " << fem_dynamic->_young << ","
			<< "\"sub_step\" : " << fem_dynamic->_sub_iteration << ","
			<< "\"density\" : " << fem_dynamic->_density <<
			"}";
	}
private:
	FEM_Dynamic* fem_dynamic;
};


class XPBD_FEM_Dynamic_Recorder : public Recorder {
public:
	virtual void init(Entity* entity) override {
		fem_dynamic = entity->get_component<XPBD_FEM_Dynamic>();

	}

	virtual void print() override {
	}

	virtual std::string get_name() override {
		return "XPBD_FEM_Dynamic";
	}
		
	virtual void add_data_json(std::ofstream& json) override {
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
private:
	XPBD_FEM_Dynamic* fem_dynamic;
};

class Mesh_VTK_Recorder : public Recorder {
public:
	Mesh_VTK_Recorder(std::string file_name) : _file_name(file_name), _mesh(nullptr) { }

	virtual void init(Entity* entity) override {
		_mesh = entity->get_component<Mesh>();
	}

	virtual void print() override { }

	virtual std::string get_name() override {
		return "vtk_file";
	}

	virtual void add_data_json(std::ofstream& json) override {
		json << "\"" << AppInfo::PathToAssets() + _file_name + ".vtk\"";
	}

	void save() override {
		VTK_Formater vtk;
		vtk.open(_file_name);
		vtk.save_mesh(_mesh->geometry(), _mesh->topologies());
		vtk.close();
	}
private:
	std::string _file_name;
	Mesh* _mesh;
};



class FEM_VTK_Recorder : public Recorder {
public:
	FEM_VTK_Recorder(std::string file_name) : _file_name(file_name), _mesh(nullptr), _ps(nullptr) { }

	virtual void init(Entity* entity) override {
		_fem_dynamic = entity->get_component<FEM_Dynamic>();
		_mesh = entity->get_component<Mesh>();
		assert(_fem_dynamic && _mesh);
		_ps = _fem_dynamic->getParticleSystem();
	}

	virtual void print() override { }

	virtual std::string get_name() override {
		return "vtk_file";
	}

	virtual void add_data_json(std::ofstream& json) override {
		json << "\"" << AppInfo::PathToAssets() + _file_name + ".vtk\"";
	}

	void save() override {
		// get particle saved data
		int nb = _ps->nb_particles();
		std::vector<Vector3> displacements(nb);
		std::vector<Vector3> init_pos(nb);
		std::vector<scalar> massses(nb);
		for (int i = 0; i < nb; ++i) {
			Particle* p = _ps->get(i);
			init_pos[i] = p->init_position;
			displacements[i] = p->position - p->init_position;
			massses[i] = p->mass;
		}

		// get element saved data
		std::vector<scalar> stress = _fem_dynamic->get_stress();
		std::vector<scalar> volume = _fem_dynamic->get_volume();
		std::vector<scalar> volume_diff = _fem_dynamic->get_volume_diff();
		std::vector<scalar> smooth_stress = _fem_dynamic->get_stress_vertices();
		

		VTK_Formater vtk;
		vtk.open(_file_name);
		vtk.save_mesh(init_pos, _mesh->topologies());
		vtk.start_point_data();
		vtk.add_scalar_data(massses, "weights");
		vtk.add_vector_data(displacements, "u");
		vtk.add_scalar_data(smooth_stress, "smooth_stress");
		vtk.start_cell_data();
		vtk.add_scalar_data(stress, "stress");
		vtk.add_scalar_data(volume, "volume");
		vtk.add_scalar_data(volume_diff, "volume_diff");
		vtk.close();
	}
protected:
	std::string _file_name;
	ParticleSystem* _ps;
	FEM_Dynamic* _fem_dynamic;
	Mesh* _mesh;
};

class Graphic_VTK_Recorder : public Recorder {
public:
	Graphic_VTK_Recorder(std::string file_name) : _file_name(file_name), _graphic(nullptr) { }

	virtual void init(Entity* entity) override {
		_graphic = entity->get_component<GL_Graphic>();
	}

	virtual void print() override { }

	virtual std::string get_name() override {
		return "vtk_graphic";
	}

	virtual void add_data_json(std::ofstream& json) override {
		json << "\"" << AppInfo::PathToAssets() + _file_name + "_Graphic" + ".vtk\"";
	}

	void save() override {
		
		std::map<Element, Mesh::Topology> topologies;
		topologies[Line] = _graphic->get_lines();
		VTK_Formater vtk;
		vtk.open(_file_name + "_Graphic");
		vtk.save_mesh(_graphic->get_geometry(), topologies);
		vtk.close();
	}
private:
	std::string _file_name;
	GL_Graphic* _graphic;
};


class FEM_Flexion_error_recorder : public Recorder {
public:
	FEM_Flexion_error_recorder(Vector3 p_follow, Vector3 p_target) : _p_follow(p_follow), _p_target(p_target) {

	}

	virtual void init(Entity* entity) override {

		
		auto fem_dynamic = entity->get_component<FEM_Dynamic>();
		assert(fem_dynamic != nullptr);
		_ps = fem_dynamic->getParticleSystem();
		
		
		bool found = false;
		for (int i = 0; i < _ps->nb_particles(); ++i) {
			Particle* p = _ps->get(i);

			if (glm::length2(p->init_position - _p_follow) >= 1e-6) continue;
			found = true;
			p_id = i;
			break;
		}

		if (!found) {
			std::cout << "ERROR : No particle found for flexion error" << std::endl;
		}
		else {
			std::cout << "Flexion Error : Follow " << p_id << "th particle" << std::endl;
		}
	}

	virtual void print() override {
		Particle* p = _ps->get(p_id);
		std::cout << "[f_err = " << glm::length2(_p_target - p->position) << "]";
	}

	virtual std::string get_name() override {
		return "flexion_error";
	}

	virtual  void add_data_json(std::ofstream& json) override {
		Particle* p = _ps->get(p_id);
		json <<
			"{"
			<< "\"error\" : " << glm::length2(_p_target - p->position) << ","
			<< "\"p\" : [" << p->position.x << ", " << p->position.y << ", " << p->position.z << "],"
			<< "\"target\" : [" << _p_target.x << ", " << _p_target.y << ", " << _p_target.z << "]"
			<<
			"}";
	}

private:
	int p_id;
	Vector3 _p_follow, _p_target;
	ParticleSystem* _ps;
};


class FEM_Torsion_error_recorder : public Recorder {
public:
	FEM_Torsion_error_recorder(scalar max_rotation, scalar beam_length)
		: _max_rotation(max_rotation), _beam_length(beam_length)
	{ }

	virtual void init(Entity* entity) override {
		{
			auto fem_dynamic = entity->get_component<XPBD_FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}
		{
			auto fem_dynamic = entity->get_component<FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}

		for (int i = 0; i < _ps->nb_particles(); ++i) {
			Particle* p = _ps->get(i);
			if (p->position.y < 1e-6 && p->position.z < 1e-6) p_ids.push_back(i);
			//if (p->position.y < 1e-6 && p->position.z > 0.9999) p_ids.push_back(i);
			//if (p->position.y > 0.9999 && p->position.z < 1e-6) p_ids.push_back(i);
			//if (p->position.y > 0.9999f && p->position.z > 0.9999) p_ids.push_back(i);
		}

		if (p_ids.size() == 0) {
			std::cout << "ERROR : No particle found for torsion error" << std::endl;
		}
		else {
			std::cout << "Flexion Error : Follow " << p_ids.size() << " particles" << std::endl;
		}
	}

	virtual void print() override {
		std::cout << "[f_err = " << "" << "]";
	}

	virtual std::string get_name() override {
		return "rotation_error";
	}

	void compute_errors(
		std::vector<scalar>& dist, 
		std::vector<scalar>& angles)
	{
		for (int i = 0; i < p_ids.size(); ++i) {
			Particle* p = _ps->get(p_ids[i]);
			Vector2 d_init = glm::normalize(Vector2(p->init_position.y, p->init_position.z) -Vector2(0.5, 0.5));
			Vector2 d_current = glm::normalize(Vector2(p->position.y, p->position.z) - Vector2(0.5, 0.5));

			scalar angle = std::abs(std::atan2(d_current.x * d_init.y - d_current.y * d_init.x, d_current.x * d_init.x + d_current.y * d_init.y));
			angle = glm::degrees(angle);
			dist.push_back(p->init_position.x / _beam_length);
			angles.push_back(angle);
		}
		for (int i = 0; i < dist.size()-1; ++i) {
			for (int j = 0; j < dist.size()-1; ++j) {
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
			temp_angle.push_back(angles[i] / j);
			temp_dist.push_back(dist[i]);
			i += j;
		}
		dist = temp_dist;
		angles = temp_angle;
	}

	virtual  void add_data_json(std::ofstream& json) override {
		std::vector<scalar> dist;
		std::vector<scalar> angles;
		compute_errors(dist, angles);
		json <<
			"{"
			<< "\"rot_max\" : " << _max_rotation << ","
			<< "\"beam_lenght\" : " << _beam_length << ","
			<< "\"distance\" : "; add_scalar_array(json, dist); json << ",";
		json << "\"angles\" : "; add_scalar_array(json, angles); json;
		json << 
			"}";
	}

	void add_scalar_array(std::ofstream& json, std::vector<scalar>& s_array) 
	{
		json << "[";
		for (int i = 0; i < s_array.size(); ++i) {
			json << s_array[i];
			if (i < s_array.size() - 1) json << ",";
		}
		json << "]";
	}

private:
	std::vector<int> p_ids;
	ParticleSystem* _ps;
	scalar _max_rotation, _beam_length;
};


