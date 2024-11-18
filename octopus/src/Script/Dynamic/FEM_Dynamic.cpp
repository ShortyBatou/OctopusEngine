#include "Script/Dynamic/FEM_Dynamic.h"
#include "Manager/TimeManager.h"
#include <iostream>

void FEM_Dynamic::update() {
    this->_ps->step(Time::Fixed_DeltaTime());
    for (int i = 0; i < _mesh->nb_vertices(); ++i) {
        _mesh->geometry()[i] = this->_ps->get(i)->position;
    }
}

ParticleSystem *FEM_Dynamic::build_particle_system() {
    return new FEM_System(new EulerSemiExplicit(0.999f), _sub_iteration);
}

std::vector<FEM_Generic *> FEM_Dynamic::build_element(const std::vector<int> &ids, Element type, scalar &volume) {
    FEM_Generic_Force *fem = new FEM_Generic_Force(ids, get_fem_material(_material, _young, _poisson),
                                                   get_fem_shape(type));
    dynamic_cast<FEM_System *>(_ps)->add_fem(fem);
    volume = fem->compute_volume(fem->get_particles(_ps->particles()));
    return std::vector<FEM_Generic *>(1, fem);
}

void FEM_Dynamic::get_fem_info(int &nb_elem, int &elem_vert, Element &elem) {
    for (auto &topo: _mesh->topologies()) {
        if (topo.second.empty()) continue;
        elem = topo.first;
        elem_vert = elem_nb_vertices(elem);
        nb_elem = static_cast<int>(topo.second.size()) / elem_vert;
    }
}

std::map<Element, std::vector<scalar> > FEM_Dynamic::get_stress() {
    std::map<Element, std::vector<scalar> > e_stress;
    for (auto &it: e_fems) {
        Element type = it.first;
        int nb_elem = static_cast<int>(_mesh->topology(type).size()) / elem_nb_vertices(type);
        std::vector<FEM_Generic *> &fems = it.second;
        std::vector<int> &id_fems = e_id_fems[type];
        std::vector<scalar> stress(nb_elem);
        std::vector<int> count(nb_elem, 0);
        for (size_t i = 0; i < fems.size(); ++i) {
            std::vector<Vector3> p = _mesh->get_elem_vertices(type, id_fems[i]);
            stress[id_fems[i]] = fems[i]->compute_stress(p);
            count[id_fems[i]]++;
        }
        for (size_t i = 0; i < stress.size(); ++i) {
            stress[i] /= static_cast<scalar>(count[i]);
        }
        e_stress[type] = stress;
    }
    return e_stress;
}

std::map<Element, std::vector<scalar> > FEM_Dynamic::get_volume() {
    std::map<Element, std::vector<scalar> > e_volume;
    for (auto &it: e_fems) {
        Element type = it.first;
        const int nb_elem = static_cast<int>(_mesh->topology(type).size()) / elem_nb_vertices(type);
        std::vector<FEM_Generic *> &fems = it.second;
        std::vector<int> &id_fems = e_id_fems[type];
        std::vector<scalar> volumes(nb_elem);
        std::vector<int> count(nb_elem, 0);
        for (size_t i = 0; i < fems.size(); ++i) {
            std::vector<Vector3> p = _mesh->get_elem_vertices(type, id_fems[i]);
            volumes[id_fems[i]] = fems[i]->compute_volume(p);
            count[id_fems[i]]++;
        }
        for (size_t i = 0; i < volumes.size(); ++i) {
            volumes[i] /= static_cast<scalar>(count[i]);
        }
        e_volume[type] = volumes;
    }

    return e_volume;
}

std::map<Element, std::vector<scalar> > FEM_Dynamic::get_volume_diff() {
    std::map<Element, std::vector<scalar> > e_volume;
    for (auto &it: e_fems) {
        Element type = it.first;
        const int nb_elem = static_cast<int>(_mesh->topology(type).size()) / elem_nb_vertices(type);
        std::vector<FEM_Generic *> &fems = it.second;
        std::vector<int> &id_fems = e_id_fems[type];
        std::vector<scalar> diff_volumes(nb_elem);
        std::vector<int> count(nb_elem, 0);
        for (size_t i = 0; i < fems.size(); ++i) {
            std::vector<Vector3> p = _mesh->get_elem_vertices(type, id_fems[i]);
            diff_volumes[id_fems[i]] = abs(fems[i]->compute_volume(p) - fems[i]->get_init_volume()) / fems[i]->
                                       get_init_volume();
            count[id_fems[i]]++;
        }
        for (size_t i = 0; i < diff_volumes.size(); ++i) {
            diff_volumes[i] /= static_cast<scalar>(count[i]);
        }
        e_volume[type] = diff_volumes;
    }

    return e_volume;
}

std::vector<scalar> FEM_Dynamic::get_masses() {
    const int n = _ps->nb_particles();
    std::vector<scalar> masses(n);
    for (int i = 0; i < n; ++i) {
        Particle *p = _ps->get(i);
        masses[i] = p->mass;
    }
    return masses;
}

std::vector<scalar> FEM_Dynamic::get_massses_inv() {
    const int n = _ps->nb_particles();
    std::vector<scalar> inv_masses(n);
    for (int i = 0; i < n; ++i) {
        Particle *p = _ps->get(i);
        inv_masses[i] = p->inv_mass;
    }
    return inv_masses;
}

std::vector<scalar> FEM_Dynamic::get_velocity_norm() {
    const int n = _ps->nb_particles();
    std::vector<scalar> velocities(n);
    for (int i = 0; i < n; ++i) {
        Particle *p = _ps->get(i);
        velocities[i] = glm::length(p->velocity);
    }
    return velocities;
}

std::vector<scalar> FEM_Dynamic::get_displacement_norm() {
    const int n = _ps->nb_particles();
    std::vector<scalar> displacement(n);
    for (int i = 0; i < n; ++i) {
        Particle *p = _ps->get(i);
        displacement[i] = glm::length(p->position - p->init_position);
    }
    return displacement;
}

std::vector<scalar> FEM_Dynamic::get_stress_vertices() {
    std::vector<scalar> smooth_stress(_ps->nb_particles(), 0.);
    for (auto &it: e_fems) {
        Element type = it.first;
        const int elem_vert = elem_nb_vertices(type);
        std::vector<FEM_Generic *> &fems = it.second;
        std::vector<int> &id_fems = e_id_fems[type];

        for (size_t i = 0; i < fems.size(); ++i) {
            Mesh::Geometry p = _mesh->get_elem_vertices(type, id_fems[i]);
            std::vector<int> t = _mesh->get_elem_indices(type, id_fems[i]);
            const scalar stress = fems[i]->compute_stress(p);
            std::vector<float> weights(elem_vert);
            for (int j = 0; j < elem_vert; ++j) {
                smooth_stress[t[j]] += stress / static_cast<scalar>(elem_vert);
            }
        }
    }

    return smooth_stress;
}


void FEM_Dynamic::build_dynamic() {
    for (Particle *p: _ps->particles()) {
        p->mass = 0;
    }

    for (auto &[e, topo]: _mesh->topologies()) {
        const std::vector<scalar> e_masses = compute_fem_mass(e, _mesh->geometry(), topo, _density, _m_distrib); // depends on density
        for(int i = 0; i < e_masses.size(); i++)
            _ps->get(i)->mass += e_masses[i];
    }

    scalar total_mass = 0.f;
    for (Particle *p: _ps->particles()) {
        p->inv_mass = 1.f / p->mass;
        total_mass += p->mass;
    }
    std::cout << "FEM TOTAL MASS = " << total_mass << std::endl;
}
