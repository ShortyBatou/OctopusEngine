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




struct FEM_Dynamic_Generic : ParticleSystemDynamic {
    FEM_Dynamic_Generic(const scalar density, const Mass_Distribution m_distrib, const scalar young,const scalar poisson,const Material material,const int sub_iteration = 30)
        : ParticleSystemDynamic(density), _density(density),_m_distrib(m_distrib), _young(young), _poisson(poisson), _sub_iteration(sub_iteration), _material(material) {
    }

    void update() override;

    ParticleSystem* build_particle_system() override;

    virtual std::vector<FEM_Generic*> build_element(const std::vector<int>& ids, Element type, scalar& volume);

    virtual void get_fem_info(int& nb_elem, int& elem_vert, Element& elem);

    virtual std::map < Element, std::vector<scalar>> get_stress();

    virtual std::map<Element, std::vector<scalar>> get_volume();

    virtual std::map<Element, std::vector<scalar>> get_volume_diff();

    virtual std::vector<scalar> get_masses();

    virtual std::vector<scalar> get_massses_inv();

    virtual std::vector<scalar> get_velocity_norm();

    virtual std::vector<scalar> get_displacement_norm();

    virtual std::vector<scalar> get_stress_vertices();

    void build_dynamic() override;

    std::map<Element, std::vector<FEM_Generic*>> e_fems;
    std::map<Element, std::vector<int>> e_id_fems;
    scalar _density;
    Mass_Distribution _m_distrib;
    scalar _young, _poisson;
    int _sub_iteration;
    Material _material;
};