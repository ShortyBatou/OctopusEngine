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

	MeshRecorder() { }

	virtual void init(Entity* entity) override {
		_mesh = entity->getComponent<Mesh>();
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

class XPBD_FEM_Dynamic_Recorder : public Recorder {
public:
	virtual void init(Entity* entity) override {
		fem_dynamic = entity->getComponent<XPBD_FEM_Dynamic>();
	}

	virtual void print() override {
		std::cout << "[XPBD_COST = " << fem_dynamic->mean_cost << "ms]";
	}

	virtual std::string get_name() override {
		return "XPBD_FEM_Dynamic";
	}
		
	virtual void add_data_json(std::ofstream& json) override {
		json <<
			"{"
			<< "\"mean_cost\" : " << fem_dynamic->mean_cost << ","
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

class XPBD_FEM_VTK_Recorder : public Recorder {
public:
	XPBD_FEM_VTK_Recorder(std::string file_name) : _file_name(file_name) {

	}

	virtual void init(Entity* entity) override {
		{
			auto fem_dynamic = entity->getComponent<XPBD_FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}
		{
			auto fem_dynamic = entity->getComponent<FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}
		_mesh = entity->getComponent<Mesh>();
	}

	virtual void print() override { }

	virtual std::string get_name() override {
		return "vtk_file";
	}

	virtual void add_data_json(std::ofstream& json) override {
		json << "\"" << AppInfo::PathToAssets() + _file_name + ".vtk\"";
	}

	void save() override {
		unsigned int nb = _ps->nb_particles();
		std::vector<scalar> norm_displacements(nb);
		std::vector<Vector3> displacements(nb);
		std::vector<Vector3> init_pos(nb);
		std::vector<scalar> massses(nb);
		for (unsigned int i = 0; i < nb; ++i) {
			Particle* p = _ps->get(i);
			init_pos[i] = p->init_position;
			displacements[i] = p->position - p->init_position;
			massses[i] = p->mass;
		}

		VTK_Formater vtk;
		vtk.open(_file_name);
		vtk.save_mesh(init_pos, _mesh->topologies());
		vtk.start_point_data();
		vtk.add_scalar_data(massses, "weights");
		vtk.add_vector_data(displacements, "u");
		vtk.close();
	}
private:
	std::string _file_name;
	ParticleSystem* _ps;
	Mesh* _mesh;
};


class FEM_Flexion_error_recorder : public Recorder {
public:
	FEM_Flexion_error_recorder(Vector3 p_follow, Vector3 p_target) : _p_follow(p_follow), _p_target(p_target) {

	}

	virtual void init(Entity* entity) override {
		{ 
			auto fem_dynamic = entity->getComponent<XPBD_FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}
		{
			auto fem_dynamic = entity->getComponent<FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}
		
		bool found = false;
		for (unsigned int i = 0; i < _ps->nb_particles(); ++i) {
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
	unsigned int p_id;
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
			auto fem_dynamic = entity->getComponent<XPBD_FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}
		{
			auto fem_dynamic = entity->getComponent<FEM_Dynamic>();
			if (fem_dynamic != nullptr) _ps = fem_dynamic->getParticleSystem();
		}

		for (unsigned int i = 0; i < _ps->nb_particles(); ++i) {
			Particle* p = _ps->get(i);
			if (p->position.y < 1e-6 && p->position.z < 1e-6) {
				p_ids.push_back(i);
			}
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
		return "flexion_error";
	}

	void compute_errors(
		std::vector<scalar>& dist, 
		std::vector<scalar>& angles)
	{
		for (unsigned int i = 0; i < p_ids.size(); ++i) {
			Particle* p = _ps->get(p_ids[i]);
			Vector2 d_init = glm::normalize(Vector2(p->init_position.y, p->init_position.z) -Vector2(0.5, 0.5));
			Vector2 d_current = glm::normalize(Vector2(p->position.y, p->position.z) - Vector2(0.5, 0.5));

			scalar angle = std::abs(std::atan2(d_current.x * d_init.y - d_current.y * d_init.x, d_current.x * d_init.x + d_current.y * d_init.y));
			angle = glm::degrees(angle);
			dist.push_back(p->init_position.x / _beam_length);
			angles.push_back(angle);
		}
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
		for (unsigned int i = 0; i < s_array.size(); ++i) {
			json << s_array[i];
			if (i < s_array.size() - 1) json << ",";
		}
		json << "]";
	}

private:
	std::vector<unsigned int> p_ids;
	ParticleSystem* _ps;
	scalar _max_rotation, _beam_length;
};


