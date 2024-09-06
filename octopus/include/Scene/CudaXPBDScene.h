#pragma once
#include <Script/Dynamic/Cuda_Constraint_Rigid_Controller.h>

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

#include "UI/UI_Component.h"
#include "Script/Dynamic/Cuda_XPBD_FEM_Dynamic.h"

struct Cuda_XPBD_Scene final : Scene
{
    char* name() override { return "Cuda XPBD Scene"; }

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
        args.young = 1e5;
        args.poisson = 0.45;
        args.damping = 0.1;
        args.iteration = 50;
        args.scenario_1 = 0;
        args.scenario_2 = 0;
        args.dir = Unit3D::right();

        const Vector3 size(4, 1, 1);
        Vector3I cells(64, 16, 16);
        build_obj(Vector3(0,0,0), cells,size, ColorBase::Black(), Tetra10, args);
    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Element element) {
        BeamMeshGenerator* generator = nullptr;
        switch (element)
        {
            case Tetra: generator = new TetraBeamGenerator(cells, size); break;
            case Pyramid: generator = new PyramidBeamGenerator(cells, size); break;
            case Prism: generator = new PrismBeamGenerator(cells, size); break;
            case Hexa: generator = new HexaBeamGenerator(cells, size); break;
            case Tetra10: generator = new TetraBeamGenerator(cells, size); break;
            case Tetra20: generator = new TetraBeamGenerator(cells, size); break;
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
        mesh->set_dynamic_geometry(true);
        return mesh;
    }

    GL_Graphic* build_graphic(const Color& color, const Element element) {
        return new GL_GraphicSurface(color);
    }

    GL_DisplayMesh* build_display() {
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->surface() = true;
        display->wireframe() = false;
        display->point() = false;
        return display;
    }

    void build_obj(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args) {
        Entity* e = Engine::CreateEnity();
        e->add_behaviour(build_beam_mesh(pos, cells, size, element));
        e->add_component(new Cuda_XPBD_FEM_Dynamic(args.density, args.young, args.poisson, args.iteration, args.damping));
        if(args.scenario_1!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + args.dir * 0.01f, -args.dir, args.scenario_1));
        if(args.scenario_2!=-1) e->add_component(new Cuda_Constraint_Rigid_Controller(pos + size - args.dir * 0.01f , args.dir, args.scenario_2 ));
        e->add_component(build_graphic(color, element));
        e->add_component(build_display());
    }
};
