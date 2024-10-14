#include "Script/Dynamic/VBD_FEM_Dynamic.h"
#include <Core/Entity.h>
#include <Manager/TimeManager.h>
#include <Mesh/Generator/BeamGenerator.h>

#include "Dynamic/Base/Solver.h"
#include "Dynamic/FEM/FEM_ContinuousMaterial.h"

ParticleSystem * VBD_FEM_Dynamic::build_particle_system()
{
    vbd = new VertexBlockDescent(new EulerSemiExplicit(1.f - _damping), _iteration, _sub_iteration, 0.93f);
    return vbd;
}

void VBD_FEM_Dynamic::build_dynamic()
{
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        // récupérer la masse
        const std::vector<scalar> masses = compute_fem_mass(e, _mesh->geometry(),topo, _density); // depends on density
        for(int i = 0; i < masses.size(); i++) {
            _ps->get(i)->mass = masses[i];
            _ps->get(i)->inv_mass = 1.f / masses[i];
        }
        vbd->addFEM(new VBD_FEM(topo, _mesh->geometry(), get_fem_shape(e), get_fem_material(_material, _young, _poisson), _damping));
        break;
    }
}

void VBD_FEM_Dynamic::update() {
    vbd->step(Time::Fixed_DeltaTime());
    update_mesh();
}
