#include "Script/Dynamic/FEM_Dynamic.h"
#include "Manager/TimeManager.h"
#include <iostream>
#include <Dynamic/FEM/FEM.h>


std::vector<scalar> FEM_Dynamic_Getters::get_residual_norm()
{
    const std::vector<Vector3> r = get_residual_vertices();
    std::vector<scalar> residual_norm(r.size());
    for (int i = 0; i < r.size(); i++) {
        residual_norm[i] = glm::length(r[i]);
    }
    return residual_norm;
}

std::vector<scalar> FEM_Dynamic::get_stress_vertices()
{

    std::vector<scalar> smooth_stress(_ps->nb_particles(), 0.);
    std::map<Element, std::vector<scalar>> stresses = get_stress();
    for (auto& [type, stress] : stresses)
    {
        const int nb_vert_elem = elem_nb_vertices(type);
        std::vector<int>& topo = _mesh->topology(type);
        const int nb_elem = static_cast<int>(topo.size()) / nb_vert_elem;

        for (size_t i = 0; i < nb_elem; i++)
        {
            for (int j = 0; j < nb_vert_elem; ++j)
            {
                const int vid = topo[i * nb_vert_elem + j];
                smooth_stress[vid] += stress[i] / static_cast<scalar>(nb_vert_elem);
            }
        }
    }
    return smooth_stress;
}

void FEM_Dynamic_Generic::update()
{
    this->_ps->step(Time::Fixed_DeltaTime());
    for (int i = 0; i < _mesh->nb_vertices(); ++i)
    {
        _mesh->geometry()[i] = this->_ps->get(i)->position;
    }
}

ParticleSystem* FEM_Dynamic_Generic::build_particle_system()
{
    return new FEM_System(new EulerSemiExplicit(0.999f), _sub_iteration);
}

std::vector<FEM_Generic*> FEM_Dynamic_Generic::build_element(const std::vector<int>& ids, const Element type, scalar& volume)
{
    FEM_Generic_Force* fem = new FEM_Generic_Force(ids, get_fem_material(_material, _young, _poisson),
                                                   get_fem_shape(type));
    dynamic_cast<FEM_System*>(_ps)->add_fem(fem);
    volume = fem->compute_volume(fem->get_particles(_ps->particles()));
    return std::vector<FEM_Generic*>(1, fem);
}

void FEM_Dynamic_Generic::get_fem_info(int& nb_elem, int& elem_vert, Element& elem)
{
    for (auto& [e, topo] : _mesh->topologies())
    {
        if (topo.empty()) continue;
        elem = e;
        elem_vert = elem_nb_vertices(elem);
        nb_elem = static_cast<int>(topo.size()) / elem_vert;
    }
}

std::map<Element, std::vector<scalar>> FEM_Dynamic_Generic::get_stress()
{
    std::map<Element, std::vector<scalar>> e_stress;
    for (auto& [type, fems] : e_fems)
    {
        const int nb_elem = static_cast<int>(_mesh->topology(type).size()) / elem_nb_vertices(type);
        std::vector<int>& id_fems = e_id_fems[type];
        std::vector<scalar> stress(nb_elem);
        std::vector<int> count(nb_elem, 0);
        for (size_t i = 0; i < fems.size(); ++i)
        {
            stress[id_fems[i]] = fems[i]->compute_stress(_mesh->get_elem_vertices(type, id_fems[i]));
            count[id_fems[i]]++; // in the case that we have multiple fem_generic per element
        }
        for (size_t i = 0; i < stress.size(); ++i)
        {
            stress[i] /= static_cast<scalar>(count[i]);
        }
        e_stress[type] = stress;
    }
    return e_stress;
}

std::map<Element, std::vector<scalar>> FEM_Dynamic_Generic::get_volume()
{
    std::map<Element, std::vector<scalar>> e_volume;
    for (auto& [e, fems] : e_fems)
    {
        const int nb_elem = static_cast<int>(_mesh->topology(e).size()) / elem_nb_vertices(e);
        std::vector<int>& id_fems = e_id_fems[e];
        std::vector<scalar> volumes(nb_elem);
        std::vector<int> count(nb_elem, 0);
        for (size_t i = 0; i < fems.size(); ++i)
        {
            std::vector<Vector3> p = _mesh->get_elem_vertices(e, id_fems[i]);
            volumes[id_fems[i]] = fems[i]->compute_volume(p);
            count[id_fems[i]]++; // in the case that we have multiple fem_generic per element
        }
        for (size_t i = 0; i < volumes.size(); ++i)
        {
            volumes[i] /= static_cast<scalar>(count[i]);
        }
        e_volume[e] = volumes;
    }

    return e_volume;
}

std::map<Element, std::vector<scalar>> FEM_Dynamic_Generic::get_volume_diff()
{
    std::map<Element, std::vector<scalar>> e_volume;
    for (auto& it : e_fems)
    {
        Element type = it.first;
        const int nb_elem = static_cast<int>(_mesh->topology(type).size()) / elem_nb_vertices(type);
        std::vector<FEM_Generic*>& fems = it.second;
        std::vector<int>& id_fems = e_id_fems[type];
        std::vector<scalar> diff_volumes(nb_elem);
        std::vector<int> count(nb_elem, 0);
        for (size_t i = 0; i < fems.size(); ++i)
        {
            std::vector<Vector3> p = _mesh->get_elem_vertices(type, id_fems[i]);
            diff_volumes[id_fems[i]] = abs(fems[i]->compute_volume(p) - fems[i]->get_init_volume()) / fems[i]->
                get_init_volume();
            count[id_fems[i]]++; // in the case that we have multiple fem_generic per element
        }
        for (size_t i = 0; i < diff_volumes.size(); ++i)
        {
            diff_volumes[i] /= static_cast<scalar>(count[i]);
        }
        e_volume[type] = diff_volumes;
    }

    return e_volume;
}



void FEM_Dynamic_Generic::build_dynamic()
{
    for (Particle* p : _ps->particles())
    {
        p->mass = 0;
    }

    for (auto& [e, topo] : _mesh->topologies())
    {
        // get mass
        const std::vector<scalar> e_masses = compute_fem_mass(e, _mesh->geometry(), topo, _density, _m_distrib);
        for (int i = 0; i < e_masses.size(); i++)
            _ps->get(i)->mass += e_masses[i];

        // build fem objects
        const int nb = elem_nb_vertices(e);
        std::vector<int> ids(nb);
        for (int i = 0; i < topo.size(); i += nb)
        {
            for (int j = 0; j < nb; ++j)
            {
                ids[j] = topo[i + j];
            }

            scalar volume = 0;
            std::vector<FEM_Generic*> fems = build_element(ids, e, volume);
            for (FEM_Generic* fem : fems)
            {
                e_fems[e].push_back(fem);
                e_id_fems[e].push_back(i / nb);
            }
        }
    }

    // build mass
    scalar total_mass = 0.f;
    for (Particle* p : _ps->particles())
    {
        p->inv_mass = 1.f / p->mass;
        total_mass += p->mass;
    }
    std::cout << "FEM TOTAL MASS = " << total_mass << std::endl;
}
