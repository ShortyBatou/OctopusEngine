#include "Script/Dynamic/MG_VBD_FEM_Dynamic.h"
#include "Dynamic/VBD/MG_VertexBlockDescent.h"
#include "Dynamic/VBD/VBD_FEM.h"
#include "Dynamic/Base/Solver.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"

ParticleSystem * MG_VBD_FEM_Dynamic::build_particle_system()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        if((e == Tetra10 || e==Hexa27)) {
            vbd = new MG_VertexBlockDescent(new EulerSemiExplicit(1.f - _damping), _iteration, _sub_iteration, _rho);
        }
        else {
            vbd = new VertexBlockDescent(new EulerSemiExplicit(1.f - _damping), _iteration, _sub_iteration, _rho);
        }
        return vbd;
    }
    return nullptr;
}


void MG_VBD_FEM_Dynamic::build_dynamic()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        // récupérer la masse
        const std::vector<scalar> masses = compute_fem_mass(e, _mesh->geometry(),topo, _density, _m_distrib); // depends on density
        for(int i = 0; i < masses.size(); i++) {
            _ps->get(i)->mass = masses[i];
            _ps->get(i)->inv_mass = 1.f / masses[i];
        }
        if(e == Tetra10 || e==Hexa27) {
            auto* mg_vbd = dynamic_cast<MG_VertexBlockDescent*>(vbd);
            auto* mg_fem = new MG_VBD_FEM(topo, _mesh->geometry(), e, get_fem_material(_material, _young, _poisson), _damping, _density, _m_distrib, _linear, _iteration);
            fem = mg_fem;
            mg_vbd->add_fem(mg_fem);
        }
        else {
            fem = new VBD_FEM(topo, _mesh->geometry(), e, get_fem_material(_material, _young, _poisson), _damping);
            vbd->add(fem);
        }
        break;
    }
}
