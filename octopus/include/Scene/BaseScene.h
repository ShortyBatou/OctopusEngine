#pragma once
#include <Script/Dynamic/Cuda_Constraint_Rigid_Controller.h>
#include <Script/Dynamic/Cuda_VBD_FEM_Dynamic.h>
#include <Script/Dynamic/XPBD_ShapeMatching_Dynamic.h>

#include "Scene.h"
#include "Manager/TimeManager.h"
#include "Manager/OpenglManager.h"
#include "Manager/CameraManager.h"
#include "Manager/InputManager.h"
#include "Manager/DebugManager.h"
#include "Manager/Dynamic.h"

#include "Mesh/Mesh.h"
#include "Mesh/Generator/BeamGenerator.h"

#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_DisplayMode.h"
#include "Rendering/GL_GraphicSurface.h"
#include "Rendering/GL_GraphicHighOrder.h"

#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "UI/UI_Component.h"

#include "Script/Dynamic/Cuda_XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/VBD_FEM_Dynamic.h"

struct SimulationArgs {
    scalar density;
    Mass_Distribution distribution;
    scalar young;
    scalar poisson;
    scalar damping;
    Material material;
    int iteration;
    int sub_iteration;
    int scenario_1;
    int scenario_2;
    Vector3 dir;
};

struct BaseScene final : Scene
{
    char* name() override { return "Basic Scene"; }

    void build_editor(UI_Editor* editor) override {
        editor->add_manager_ui(new UI_Time());
        editor->add_manager_ui(new UI_Dynamic());
        editor->add_manager_ui(new UI_DisplaySettings());
        editor->add_manager_ui(new UI_Camera());
        editor->add_component_ui(new UI_Mesh_Display());
        editor->add_component_ui(new UI_Cuda_Constraint_Rigid_Controller());
        editor->add_component_ui(new UI_Data_Recorder());
        editor->add_component_ui(new UI_Graphic_Saver());
    }

    void build_root(Entity* root) override
    {
        root->add_behaviour(new TimeManager(1.f / 60.f));
        root->add_behaviour(new DynamicManager(Vector3(0.,-9.81*1.f,0.)));
        root->add_behaviour(new InputManager());
        root->add_behaviour(new CameraManager());
        root->add_behaviour(new DebugManager(true));
        root->add_behaviour(new OpenGLManager(Color(0.9f,0.9f,0.9f,1.f)));
    }

    // build scene's entities
    void build_entities() override
    {  
        SimulationArgs args{};
        args.density = 1000;
        args.distribution = Shape;
        args.young = 1e7f;
        args.poisson = 0.49;
        args.damping = 1e-5;
        args.iteration = 40;
        args.sub_iteration = 1;
        args.scenario_1 = 0;
        args.scenario_2 = -1;
        args.dir = Unit3D::right();
        args.material = Stable_NeoHooke;

        const Vector3 size(2, 1, 1);
        Vector3I cells(16, 8, 8);
        //(Vector3(0,0,0), cells,size, ColorBase::Red(), Hexa, args);
        //build_xpbd_entity(Vector3(0,0,0),cells, size, Color(0.3,0.8,0.3,0.), Tetra, args, false);
        //build_xpbd_entity(Vector3(0,0,1.2),cells, size, Color(0.4,0.4,0.8,0.), Tetra10, args, true, false);
        build_vbd_entity(Vector3(0,0,0),cells, size, Color(0.3,0.3,0.8,0.), Tetra, args, true);
        //build_vbd_entity(Vector3(0,0,1),cells, size, Color(0.3,0.8,0.3,0.), Hexa27, args, false);
        args.iteration = 100;
        //build_vbd_entity(Vector3(0,0,2),cells, size, Color(0.3,0.8,0.8,0.), Hexa27, args, false);

        args.damping = 5e-6;
        args.sub_iteration = 300;
        //build_vbd_entity(Vector3(0,0,0),cells, size, Color(0.8,0.3,0.8,0.), Tdetra10, args, true);
        //build_fem_entity(Vector3(0,0,-1),cells, size, Color(0.8,0.8,0.8,0.), Hexa27, args, true);
        args.damping = 1e-7;
        build_fem_entity(Vector3(0,0,0),cells, size, Color(0.3,0.8,0.8,0.), Hexa, args, true);

        //build_fem_entity(Vector3(0,0,1),cells, size, Color(0.8,0.8,0.8,0.), Hexa27, args, true);

        args.sub_iteration = 50;
        args.iteration = 1;
        //build_vbd_entity(Vector3(0,0,0),cells, size, Color(0.8,0.3,0.8,0.), Hexa, args, true);
        args.damping = 1e-6;
        //build_fem_entity(Vector3(0,0,0),cells, size, Color(0.8,0.3,0.8,0.), Hexa, args, false);

        //build_vbd_entity(Vector3(0,0,0),cells, size, Color(0.8,0.3,0.8,0.), Hexa, args, true);
        args.damping = 20;
        args.sub_iteration = 100;
        //build_xpbd_entity(Vector3(0,0,1),cells, size, Color(0.7,0.4,0.8,0.), Hexa, args, true, true);
        //build_xpbd_entity(Vector3(0,0,-2.2),cells, size, Color(0.7,0.4,0.8,0.), Tetra, args, true, true);

    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Element element) {
        BeamMeshGenerator* generator = nullptr;
        switch (element)
        {
            case Tetra: case Tetra10: case Tetra20: generator = new TetraBeamGenerator(cells, size); break;
            case Pyramid: generator = new PyramidBeamGenerator(cells, size); break;
            case Prism: generator = new PrismBeamGenerator(cells, size); break;
            case Hexa: case Hexa27: generator = new HexaBeamGenerator(cells, size); break;
            default: generator = new TetraBeamGenerator(cells, size); break;
        }
        generator->setTransform(glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = generator->build();
        mesh->set_dynamic_topology(false);
        delete generator;
        return mesh;
    }

    Mesh* build_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, Element element) {
        Mesh* mesh = get_beam_mesh(pos, cells, size, element);
        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
        if (element == Hexa27) hexa_to_hexa27(mesh->geometry(), mesh->topologies());
        mesh->set_dynamic_geometry(true);
        return mesh;
    }

    GL_Graphic* build_graphic(const Color& color, const Element element) {
        //return new GL_GraphicHighOrder(2, color);
        return new GL_GraphicSurface(color);
    }

    GL_DisplayMesh* build_display() {
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->surface() = true;
        display->wireframe() = true;
        display->point() = false;
        return display;
    }



    void add_constraint(Entity* e, const Vector3& pos, const Vector3& size, const SimulationArgs& args) {
        // constraint for Particle system
        if (args.scenario_1 != -1) {
            const auto rd_constraint_1 = new Constraint_Rigid_Controller(Unit3D::Zero() + args.dir*0.01f, -args.dir, args.scenario_1);
            e->add_component(rd_constraint_1);
            rd_constraint_1->_rot_speed = 90;
            rd_constraint_1->_move_speed = 1.;
        }

        if (args.scenario_2 != -1) {
            auto rd_constraint_2 = new Constraint_Rigid_Controller(pos + size, args.dir, args.scenario_2);
            rd_constraint_2->_rot_speed = 90;
            rd_constraint_2->_move_speed = 1.;
            e->add_component(rd_constraint_2);
        }
    }

    void build_fem_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, bool gpu) {
        Entity* e = Engine::CreateEnity();
        e->add_behaviour(build_beam_mesh(pos, cells, size, element));
        if(gpu)
        {
            e->add_component(new Cuda_FEM_Dynamic(std::max(args.iteration, args.sub_iteration), args.density, args.distribution, args.young, args.poisson, args.material, args.damping));
            if(args.scenario_1!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + args.dir * 0.01f, -args.dir, args.scenario_1));
            if(args.scenario_2!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + size - args.dir * 0.01f , args.dir, args.scenario_2 ));
        }
        else
        {
            e->add_component(new FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material, std::max(args.iteration, args.sub_iteration)));
            add_constraint(e, pos, size, args);
        }

        e->add_component(build_graphic(color, element));
        e->add_component(build_display());
    }


    void build_vbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, bool gpu) {
        Entity* e = Engine::CreateEnity();
        e->add_behaviour(build_beam_mesh(pos, cells, size, element));
        if(gpu)
        {
            e->add_component(new Cuda_VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material, args.iteration, args.sub_iteration, args.damping));
            if(args.scenario_1!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + args.dir * 0.01f, -args.dir, args.scenario_1));
            if(args.scenario_2!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + size - args.dir * 0.01f , args.dir, args.scenario_2 ));
        }
        else
        {
            e->add_component(new VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material,args.iteration, args.sub_iteration, args.damping));
            add_constraint(e, pos, size, args);
        }

        e->add_component(build_graphic(color, element));
        e->add_component(build_display());
    }

    void build_xpbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, bool gpu, bool coupled) {
        Entity* e = Engine::CreateEnity();
        e->add_behaviour(build_beam_mesh(pos, cells, size, element));
        if(gpu) {
            e->add_component(new Cuda_XPBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson,args.material,std::max(args.iteration, args.sub_iteration), args.damping, coupled));
            if(args.scenario_1!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + args.dir * 0.01f, -args.dir, args.scenario_1));
            if(args.scenario_2!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + size - args.dir * 0.01f , args.dir, args.scenario_2 ));
        }
        else {
            e->add_component(new XPBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson,args.material, args.iteration, args.sub_iteration, args.damping, coupled));
            add_constraint(e, pos, size, args);
        }
        e->add_component(build_graphic(color, element));
        e->add_component(build_display());
    }


};
