#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"

#include "Manager/Input.h"
#include "Dynamic/FEM/FEM.h"
#include "Dynamic/Base/Constraint.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Generic_Force.h"
#include "Script/Dynamic/ParticleSystemDynamic.h"

struct FEM_Dynamic : public ParticleSystemDynamic {
    FEM_Dynamic(scalar density, scalar young, scalar poisson, Material material, int sub_iteration = 30)
        : ParticleSystemDynamic(density), _young(young), _poisson(poisson), _material(material), _sub_iteration(sub_iteration), _density(density) {
    }

    virtual void update() override {
        this->_ps->step(Time::Fixed_DeltaTime());
        for (int i = 0; i < _mesh->nb_vertices(); ++i) {
            _mesh->geometry()[i] = this->_ps->get(i)->position;
        }
    }

    virtual ParticleSystem* build_particle_system() override {
        return new FEM_System(new EulerSemiExplicit(0.999f), _sub_iteration);
    }

    virtual std::vector<FEM_Generic*> build_element(const std::vector<int>& ids, Element type, scalar& volume) {
        FEM_Generic_Force* fem = new FEM_Generic_Force(ids, get_fem_material(_material, _young, _poisson), get_fem_shape(type));
        static_cast<FEM_System*>(_ps)->add_fem(fem);
        volume = fem->compute_volume(fem->get_particles(_ps->particles()));
        return std::vector<FEM_Generic*>(1, fem);
    }

    virtual void get_fem_info(int& nb_elem, int& elem_vert, Element& elem) {
        for (auto& topo : _mesh->topologies()) {
            if (topo.second.size() == 0) continue;
            elem = topo.first;
            elem_vert = elem_nb_vertices(elem);
            nb_elem = topo.second.size() / elem_vert;
        }
    }

    virtual std::map < Element, std::vector<scalar>> get_stress()
    {       
        std::map<Element, std::vector<scalar>> e_stress;
        for (auto& it : e_fems) {
            Element type = it.first;
            int nb_elem = _mesh->topology(type).size() / elem_nb_vertices(type);
            std::vector<FEM_Generic*>& fems = it.second;
            std::vector<int>& id_fems = e_id_fems[type];
            std::vector<scalar> stress(nb_elem);
            std::vector<int> count(nb_elem, 0);
            for (size_t i = 0; i < fems.size(); ++i) {
                std::vector<Vector3> p = _mesh->get_elem_vertices(type, id_fems[i]);
                stress[id_fems[i]] = fems[i]->compute_stress(p);
                count[id_fems[i]]++;
            }
            for (size_t i = 0; i < stress.size(); ++i) {
                stress[i] /= count[i];
            }
            e_stress[type] = stress;
        }
        return e_stress;
    }

    virtual std::map<Element, std::vector<scalar>> get_volume()
    {
        std::map<Element, std::vector<scalar>> e_volume;
        for (auto& it : e_fems) {
            Element type = it.first;
            int nb_elem = _mesh->topology(type).size() / elem_nb_vertices(type);
            std::vector<FEM_Generic*>& fems = it.second;
            std::vector<int>& id_fems = e_id_fems[type];
            std::vector<scalar> volumes(nb_elem);
            std::vector<int> count(nb_elem, 0);
            for (size_t i = 0; i < fems.size(); ++i) {
                std::vector<Vector3> p = _mesh->get_elem_vertices(type, id_fems[i]);
                volumes[id_fems[i]] = fems[i]->compute_volume(p);
                count[id_fems[i]]++;
            }
            for (size_t i = 0; i < volumes.size(); ++i) {
                volumes[i] /= count[i];
            }
            e_volume[type] = volumes;
        }
        
        return e_volume;
    }

    virtual std::map<Element, std::vector<scalar>> get_volume_diff()
    {

        std::map<Element, std::vector<scalar>> e_volume;
        for (auto& it : e_fems) {
            Element type = it.first;
            int nb_elem = _mesh->topology(type).size() / elem_nb_vertices(type);
            std::vector<FEM_Generic*>& fems = it.second;
            std::vector<int>& id_fems = e_id_fems[type];
            std::vector<scalar> diff_volumes(nb_elem);
            std::vector<int> count(nb_elem, 0);
            for (size_t i = 0; i < fems.size(); ++i) {
                std::vector<Vector3> p = _mesh->get_elem_vertices(type, id_fems[i]);
                diff_volumes[id_fems[i]] = abs(fems[i]->compute_volume(p) - fems[i]->get_init_volume());
                count[id_fems[i]]++;
            }
            for (size_t i = 0; i < diff_volumes.size(); ++i) {
                diff_volumes[i] /= count[i];
            }
            e_volume[type] = diff_volumes;
        }

        return e_volume;
    }

    virtual std::vector<scalar> get_masses() {
        std::vector<scalar> masses(_ps->nb_particles());
        for (size_t i = 0; i < masses.size(); ++i) {
            Particle* p = _ps->get(i);
            masses[i] = p->mass;
        }
        return masses;
    }

    virtual std::vector<scalar> get_massses_inv() {
        std::vector<scalar> inv_masses(_ps->nb_particles());
        for (size_t i = 0; i < inv_masses.size(); ++i) {
            Particle* p = _ps->get(i);
            inv_masses[i] = p->inv_mass;
        }
        return inv_masses;
    }

    virtual std::vector<scalar> get_velocity_norm() {
        std::vector<scalar> velocities(_ps->nb_particles());
        for (size_t i = 0; i < velocities.size(); ++i) {
            Particle* p = _ps->get(i);
            velocities[i] = glm::length(p->velocity);
        }
        return velocities;
    }

    virtual std::vector<scalar> get_displacement_norm()
    {
        std::vector<scalar> displacement(_ps->nb_particles());
        for (size_t i = 0; i < displacement.size(); ++i) {
            Particle* p = _ps->get(i);
            displacement[i] = glm::length(p->position - p->init_position);
        }
        return displacement;
    }

    virtual std::vector<scalar> get_stress_vertices()
    {
        std::vector<scalar> smooth_stress(_ps->nb_particles(),0.);
        for (auto& it : e_fems) {
            Element type = it.first;
            int elem_vert = elem_nb_vertices(type);
            std::vector<FEM_Generic*>& fems = it.second;
            std::vector<int>& id_fems = e_id_fems[type];

            for (size_t i = 0; i < fems.size(); ++i) {
                Mesh::Geometry p = _mesh->get_elem_vertices(type, id_fems[i]);
                std::vector<int> t = _mesh->get_elem_indices(type, id_fems[i]);
                scalar stress = fems[i]->compute_stress(p);
                std::vector<float> weights(elem_vert);
                for (int j = 0; j < elem_vert; ++j) {
                    smooth_stress[t[j]] += stress / elem_vert;
                }
            }
        }
        
        return smooth_stress;
    }


    virtual void build_dynamic() {
        for (Particle* p : _ps->particles()) {
            p->mass = 0;
        }
        for (auto &topo : _mesh->topologies()) {
            Element type = topo.first;
            int nb = elem_nb_vertices(type);
            std::vector<int> ids(nb);
            for (int i = 0; i < topo.second.size(); i += nb) {
                for (int j = 0; j < nb; ++j) {
                    ids[j] = topo.second[i + j];
                }

                scalar volume = 0;
                std::vector<FEM_Generic*> fems = build_element(ids, type, volume);
                for (FEM_Generic* fem : fems) {
                    e_fems[type].push_back(fem);
                    e_id_fems[type].push_back(i / nb);
                }
                    
                
                for (int j = 0; j < nb; ++j) {
                    Particle* p = _ps->get(ids[j]);
                    p->mass += _density * volume / nb;
                }
            }
        }
        scalar total_mass = 0;
        for (Particle* p : _ps->particles()) {
            p->inv_mass = scalar(1) / p->mass;
            total_mass += p->mass;
        }
        std::cout << "FEM TOTAL MASS = " << total_mass << std::endl;
    }

public:
    std::map<Element, std::vector<FEM_Generic*>> e_fems;
    std::map<Element, std::vector<int>> e_id_fems;
    scalar _density;
    scalar _young, _poisson;
    int _sub_iteration;
    Material _material;
};