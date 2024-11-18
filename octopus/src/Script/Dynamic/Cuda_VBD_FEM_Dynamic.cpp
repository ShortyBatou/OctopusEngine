#include "Core/Entity.h"
#include "Manager/TimeManager.h"
#include "Tools/Color.h"
#include <Rendering/GL_Graphic.h>
#include "Script/Dynamic/Cuda_VBD_FEM_Dynamic.h"
#include "Tools/Random.h"
#include <Manager/Debug.h>
#include "GPU/GPU_PBD.h"
#include <set>
#include <Manager/Input.h>


void Cuda_VBD_FEM_Dynamic::init() {
    _mesh = _entity->get_component<Mesh>();
    std::vector masses(_mesh->nb_vertices(),0.f);
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        // récupérer la masse
        const std::vector<scalar> e_masses = compute_fem_mass(e, _mesh->geometry(),topo, _density, _m_distrib); // depends on density
        for(size_t i = 0; i < e_masses.size(); i++)
            masses[i] += e_masses[i];
    }

    vbd = new GPU_VBD(_mesh->geometry(), masses, _iteration, _sub_iteration, _damping);
    for(auto&[e, topo] : _mesh->topologies()) {
        if(topo.empty()) continue;
        vbd->dynamic = new GPU_VBD_FEM(e, topo, _mesh->geometry(), _young, _poisson);
        break;
    }

}

void Cuda_VBD_FEM_Dynamic::update() {
    if(Input::Down(Key::Q)) vbd->sub_iteration++;
    if(Input::Down(Key::A)) vbd->sub_iteration--;
    if(Input::Down(Key::W)) vbd->iteration++;
    if(Input::Down(Key::S)) vbd->iteration--;

    if(Input::Down(Key::Q) || Input::Down(Key::A) || Input::Down(Key::W) || Input::Down(Key::S)) {
        std::cout << vbd->sub_iteration << " " << vbd->iteration << std::endl;
    }


    vbd->step(Time::Fixed_DeltaTime());
    vbd->get_position(_mesh->geometry());
}

