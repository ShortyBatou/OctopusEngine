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
        return new FEM_System(new EulerSemiExplicit(0.995f), _sub_iteration);
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

    virtual std::vector<scalar> get_stress()
    {
        int nb_elem, elem_vert; Element elem;
        get_fem_info(nb_elem, elem_vert, elem);
        std::vector<scalar> stress(nb_elem);
        for (size_t i = 0; i < fems.size(); ++i) {
            std::vector<Vector3> p = _mesh->get_elem_vertices(elem, id_fems[i]);
            stress[id_fems[i]] = fems[i]->compute_stress(p);
        }
        return stress;
    }

    virtual std::vector<scalar> get_volume()
    {
        int nb_elem, elem_vert; Element elem;
        get_fem_info(nb_elem, elem_vert, elem);
        std::vector<scalar> volume(nb_elem);
        for (size_t i = 0; i < fems.size(); ++i) {
            std::vector<Vector3> p = _mesh->get_elem_vertices(elem, id_fems[i]);
            volume[id_fems[i]] = fems[i]->compute_volume(p);
        }
        return volume;
    }

    virtual std::vector<scalar> get_volume_diff()
    {
        int nb_elem, elem_vert; Element elem;
        get_fem_info(nb_elem, elem_vert, elem);
        std::vector<scalar> volume_diff(nb_elem);
        for (size_t i = 0; i < fems.size(); ++i) {
            std::vector<Vector3> p = _mesh->get_elem_vertices(elem, id_fems[i]);
            volume_diff[id_fems[i]] = abs(fems[i]->compute_volume(p) - fems[i]->get_init_volume());
        }
        return volume_diff;
    }

    virtual std::vector<scalar> get_stress_vertices()
    {
        int nb_elem, elem_vert; Element elem;
        get_fem_info(nb_elem, elem_vert, elem);
        //Mesh::Topology& topo = _mesh->topology(elem);
        //std::vector<int> v_count(_mesh->geometry().size(), 0);
        //for (size_t i = 0; i < topo.size(); ++i) {
        //    v_count[topo[i]]++;
        //}

        std::vector<scalar> smooth_stress(_ps->nb_particles(),0.);
        for (size_t i = 0; i < fems.size(); ++i) {
            Mesh::Geometry p = _mesh->get_elem_vertices(elem, id_fems[i]);
            std::vector<int> t = _mesh->get_elem_indices(elem, id_fems[i]);
            scalar stress = fems[i]->compute_stress(p);
            std::vector<float> weights(elem_vert);

            //scalar sum = 0;
            //for (int j = 0; j < elem_vert; ++j) {
            //    sum += 1.f / v_count[t[j]];
            //}

            //for (int j = 0; j < elem_vert; ++j) {
            //    smooth_stress[t[j]] += stress / (v_count[topo[j]] * sum);
            //}
            for (int j = 0; j < elem_vert; ++j) {
                smooth_stress[t[j]] += stress / elem_vert;
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
                std::vector<FEM_Generic*> e_fems = build_element(ids, type, volume);
                for (FEM_Generic* fem : e_fems) {
                    fems.push_back(fem);
                    id_fems.push_back(i / nb);
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
    std::vector<FEM_Generic*> fems;
    std::vector<int> id_fems;
    scalar _density;
    scalar _young, _poisson;
    int _sub_iteration;
    Material _material;
};