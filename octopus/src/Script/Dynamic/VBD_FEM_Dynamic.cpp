#include "Script/Dynamic/VBD_FEM_Dynamic.h"
#include <Core/Entity.h>
#include <Manager/TimeManager.h>
#include <Mesh/Generator/BeamGenerator.h>
#include <Dynamic/VBD/MG_VertexBlockDescent.h>
#include <Dynamic/VBD/VBD_FEM.h>

#include "Dynamic/Base/Solver.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"

ParticleSystem * VBD_FEM_Dynamic::build_particle_system()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        vbd = new VertexBlockDescent(new EulerSemiExplicit(1.f), _iteration, _sub_iteration, _rho);
        return vbd;
    }
    return nullptr;
}

void VBD_FEM_Dynamic::build_dynamic()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        // récupérer la masse
        const std::vector<scalar> masses = compute_fem_mass(e, _mesh->geometry(),topo, _density, _m_distrib); // depends on density
        for(int i = 0; i < masses.size(); i++) {
            _ps->get(i)->mass = masses[i];
            _ps->get(i)->inv_mass = 1.f / masses[i];
        }
        fem = new VBD_FEM(topo, _mesh->geometry(), e, get_fem_material(_material, _young, _poisson), _damping);
        vbd->add(fem);
        break;
    }
}

void VBD_FEM_Dynamic::update() {
    vbd->step(Time::Fixed_DeltaTime());
    update_mesh();
}

std::map<Element, std::vector<scalar>> VBD_FEM_Dynamic::get_stress()
{
    std::map<Element, std::vector<scalar>> stresses;
    for(auto&[e, topo] : _mesh->topologies())
    {
        if(topo.empty()) continue;
        stresses[e] = fem->compute_stress(vbd);
        break;
    }
    return stresses;
}

std::map<Element, std::vector<scalar>> VBD_FEM_Dynamic::get_volume()
{
    std::map<Element, std::vector<scalar>> volumes;
    for(auto&[e, topo] : _mesh->topologies())
    {
        if(topo.empty()) continue;
        volumes[e] = fem->compute_volume(vbd);
        break;
    }
    return volumes;
}

std::map<Element, std::vector<scalar>> VBD_FEM_Dynamic::get_volume_diff()
{
    std::map<Element, std::vector<scalar>> volumes_diff;
    for(auto&[e, topo] : _mesh->topologies())
    {
        if(topo.empty()) continue;
        volumes_diff[e] = fem->compute_colume_diff(vbd);
        break;
    }
    return volumes_diff;
}
