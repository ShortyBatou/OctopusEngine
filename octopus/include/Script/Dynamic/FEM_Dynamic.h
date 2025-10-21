#pragma once
#include "Core/Base.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"
#include "Manager/Input.h"
#include "Dynamic/FEM/FEM_Shape.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"
#include "Dynamic/FEM/FEM_Generic_Force.h"
#include "Script/Dynamic/ParticleSystemDynamic.h"

struct FEM_Dynamic_Getters
{
    virtual ~FEM_Dynamic_Getters() = default;
    [[nodiscard]] virtual std::map<Element, std::vector<scalar>> get_stress() = 0;
    [[nodiscard]] virtual std::map<Element, std::vector<scalar>> get_volume() = 0;
    [[nodiscard]] virtual std::map<Element, std::vector<scalar>> get_volume_diff() = 0;
    [[nodiscard]] virtual std::map<Element, std::vector<scalar>> get_inverted() = 0;
    [[nodiscard]] virtual std::vector<scalar> get_stress_vertices() = 0;
    [[nodiscard]] virtual std::vector<Vector3> get_residual_vertices() = 0;
    [[nodiscard]] virtual std::vector<scalar> get_residual_norm();
};

struct FEM_Dynamic : public ParticleSystemDynamic, public FEM_Dynamic_Getters
{
    FEM_Dynamic(const scalar density, const Mass_Distribution m_distrib, const scalar young,const scalar poisson,const Material material,const int sub_iteration = 30)
        : ParticleSystemDynamic(density), _density(density),_m_distrib(m_distrib), _young(young), _poisson(poisson), _sub_iteration(sub_iteration), _material(material) {
    }

    std::vector<scalar> get_stress_vertices() override;
    std::vector<Vector3> get_residual_vertices() override { return std::vector<Vector3>(_ps->nb_particles(), Vector3(0));}
    scalar _density;
    Mass_Distribution _m_distrib;
    scalar _young, _poisson;
    int _sub_iteration;
    Material _material;
};


struct FEM_Dynamic_Generic : FEM_Dynamic {
    FEM_Dynamic_Generic(const scalar density, const Mass_Distribution m_distrib, const scalar young,const scalar poisson,const Material material,const int sub_iteration = 30)
        : FEM_Dynamic(density, m_distrib, young, poisson, material, sub_iteration) {
    }

    void update() override;
    void build_dynamic() override;
    ParticleSystem* build_particle_system() override;

    [[nodiscard]] std::map<Element, std::vector<scalar>> get_stress() override;
    [[nodiscard]] std::map<Element, std::vector<scalar>> get_volume() override;
    [[nodiscard]] std::map<Element, std::vector<scalar>> get_volume_diff() override;
    [[nodiscard]] std::map<Element, std::vector<scalar>> get_inverted() override;

    virtual std::vector<FEM_Generic*> build_element(const std::vector<int>& ids, Element type, scalar& volume);
    virtual void get_fem_info(int& nb_elem, int& elem_vert, Element& elem);

    std::map<Element, std::vector<FEM_Generic*>> e_fems;
    std::map<Element, std::vector<int>> e_id_fems;
};