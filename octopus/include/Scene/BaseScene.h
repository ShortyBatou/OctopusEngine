#pragma once
#include "Scene.h"
#include "Manager/TimeManager.h"
#include "Manager/OpenglManager.h"
#include "Manager/CameraManager.h"
#include "Manager/InputManager.h"
#include "Manager/DebugManager.h"

#include "Mesh/Mesh.h"
#include "Mesh/Generator/PrimitiveGenerator.h"
#include "Mesh/Generator/BeamGenerator.h"
#include "Mesh/Converter/MeshLoader.h"

#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_GraphicElement.h"
#include "Rendering/GL_DisplayMode.h"
#include "Rendering/GL_GraphicSurface.h"
#include "Rendering/GL_GraphicHighOrder.h"

#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"
#include "Script/Dynamic/Constraint_Rigid_Controller.h"
#include "Script/Dynamic/ConstantForce_Controller.h"
#include "Script/VTK/VTK_FEM.h"
#include "Script/Record/DataRecorder.h"
#include "UI/UI_Component.h"

struct SimulationArgs {
    scalar density;
    scalar young;
    scalar poisson;
    Material material;
    int iteration;
    int sub_iteration;
    int scenario_1;
    int scenario_2;
    Vector3 dir;
};

struct BaseScene : public Scene
{
    virtual char* name() override { return "Basic Scene"; }

    virtual void init() override { 
        
    }

    virtual void build_editor(UI_Editor* editor) {
        editor->add_manager_ui(new UI_Time());
        editor->add_manager_ui(new UI_SceneColor());
        editor->add_manager_ui(new UI_Camera());
        editor->add_component_ui(new UI_Mesh());
        editor->add_component_ui(new UI_Data_Recorder());
        editor->add_component_ui(new UI_Graphic_Saver()); 
        editor->add_component_ui(new UI_PBD_Dynamic());
        editor->add_component_ui(new UI_Constraint_Rigid_Controller());
    }

    virtual void build_root(Entity* root) override
    {
        root->addBehaviour(new TimeManager(1.f / 60.f));
        root->addBehaviour(new InputManager());
        root->addBehaviour(new CameraManager());
        root->addBehaviour(new DebugManager(true));
        root->addBehaviour(new OpenGLManager(Color(0.9f,0.9f,0.9f,1.f)));
    }

    // build scene's entities
    virtual void build_entities() override 
    {  
        SimulationArgs args;
        args.density = 1000;
        args.material = StVK;
        args.poisson = 0.35;
        args.young = 3e6;
        args.iteration = 1;
        args.sub_iteration = 50;
        args.scenario_1 = 0;
        args.scenario_2 = -1;
        args.dir = Unit3D::right();

        Vector3 size(4, 1, 1);
        Vector3I cells = Vector3I(64,16,16);
        //build_xpbd_entity(Vector3(0, 0, 0), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra10, false, false);
        build_xpbd_fem_entity(Vector3(0, 0, 0), cells, size, Color(0.8f, 0.3f, 0.8f, 1.f), Tetra, args);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.3, 0.8, 0.3, 1.), Tetra20, false, false);
        //build_xpbd_entity(Vector3(0, 0, 2), cells, size, Color(0.3, 0.3, 0.8, 1.), Tetra20, false, false);
        //cells = Vector3I(8, 3, 3);
        //cells = Vector3I(6, 2, 2);
        //build_xpbd_entity(Vector3(0, 0, 2), cells, size, Color(0.8, 0.3, 0.3, 1.), Tetra, false, true);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra10, true, false);
    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, Element element) {
        BeamMeshGenerator* generator = nullptr;
        switch (element)
        {
            case Tetra: generator = new TetraBeamGenerator(cells, size); break;
            case Pyramid: generator = new PyramidBeamGenerator(cells, size); break;
            case Prism: generator = new PrismBeamGenerator(cells, size); break;
            case Hexa: generator = new HexaBeamGenerator(cells, size); break;
            case Tetra10: generator = new TetraBeamGenerator(cells, size); break;
            case Tetra20: generator = new TetraBeamGenerator(cells, size); break;
        default: break; }
        generator->setTransform(glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = generator->build();
        delete generator;
        return mesh;
    }

    Mesh* build_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, Element element) {
        Mesh* mesh;

        mesh = get_beam_mesh(pos, cells, size, element);
        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());

        mesh->set_dynamic_geometry(true);
        return mesh;
    }

    GL_Graphic* build_graphic(const Color& color, Element element) {
        // Mesh converter simulation to rendering (how it will be displayed)
        GL_Graphic* graphic;

        //graphic = new GL_GraphicElement(0.7);
        if (element == Tetra10 || element == Tetra20)
            //graphic = new GL_GraphicElement(0.7);
            graphic = new GL_GraphicHighOrder(3, color);
        else
            graphic = new GL_GraphicSurface(color);

        graphic->normals() = false;
        return graphic;
    }

    GL_DisplayMesh* build_display() {
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->wireframe() = true;
        display->point() = false;
        display->normal() = false;
        return display;
    }

    void add_constraint(Entity* e, const Vector3& pos, const Vector3& size, SimulationArgs& args) {
        // constraint for Particle system
        if (args.scenario_1 != -1) {
            auto rd_constraint_1 = new Constraint_Rigid_Controller(Unit3D::Zero(), -args.dir, args.scenario_1);
            e->addComponent(rd_constraint_1);
            rd_constraint_1->_rot_speed = 180;
            rd_constraint_1->_move_speed = 1.;
        }

        if (args.scenario_2 != -1) {
            auto rd_constraint_2 = new Constraint_Rigid_Controller(pos + size, args.dir, args.scenario_2);
            rd_constraint_2->_rot_speed = 180;
            rd_constraint_2->_move_speed = 1.;
            e->addComponent(rd_constraint_2);
        }
    }

    Mesh* build_vtk_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, std::string file) {
        VTK_Loader loader(AppInfo::PathToAssets() + file);
        loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = loader.build();
        mesh->set_dynamic_geometry(true);
        return mesh;
    }

    void build_fem_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element, SimulationArgs& args) {
        Entity* e = Engine::CreateEnity();
        e->addBehaviour(build_beam_mesh(pos, cells, size, element));
        e->addComponent(new FEM_Dynamic(args.density, args.young, args.poisson, args.material, args.iteration));
        add_constraint(e, pos, size, args);
        e->addComponent(build_graphic(color, element));
        e->addComponent(build_display());

        // save mesh in VTK format (Paraview)size
        std::string file_name = std::string(element_name(element)) + "_" + std::to_string(cells.x) + "_" + std::to_string(cells.y) + "_" + std::to_string(cells.z)
            + "_" + std::to_string(int(size.x)) + "x" + std::to_string(int(size.y)) + "x" + std::to_string(int(size.z));
        DataRecorder* data_recorder = new DataRecorder(file_name);
        data_recorder->add(new TimeRecorder());
        data_recorder->add(new MeshRecorder());
        data_recorder->add(new FEM_Dynamic_Recorder());
        data_recorder->add(new FEM_VTK_Recorder(file_name));
        data_recorder->add(new Graphic_VTK_Recorder(file_name));
        e->addComponent(data_recorder);
    }


    void build_xpbd_fem_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element, SimulationArgs& args) {
        Entity* e = Engine::CreateEnity();
        //e->addBehaviour(build_beam_mesh(pos, cells, size, element));

        Mesh* mesh = build_vtk_mesh(pos, cells, size, color, "mesh/vtk/beam-s-4-1-1-n-16-4-4-tetra.vtk");
        subdive_tetra(mesh->geometry(), mesh->topologies());
        subdive_tetra(mesh->geometry(), mesh->topologies());
        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
        e->addBehaviour(mesh);

        e->addComponent(new XPBD_FEM_Dynamic(args.density, args.young, args.poisson, args.material, args.iteration, args.sub_iteration, 1.));
        add_constraint(e, pos, size, args);
        e->addComponent(build_graphic(color, element));
        e->addComponent(build_display());

        // save mesh in VTK format (Paraview)size
        std::string file_name = std::string(element_name(element)) + "_" + std::to_string(cells.x) + "_" + std::to_string(cells.y) + "_" + std::to_string(cells.z)
            + "_" + std::to_string(int(size.x)) + "x" + std::to_string(int(size.y)) + "x" + std::to_string(int(size.z));
        DataRecorder* data_recorder = new DataRecorder(file_name);
        data_recorder->add(new TimeRecorder());
        data_recorder->add(new MeshRecorder());
        data_recorder->add(new XPBD_FEM_Dynamic_Recorder());
        data_recorder->add(new FEM_VTK_Recorder(file_name));
        data_recorder->add(new Graphic_VTK_Recorder(file_name));
        //data_recorder->add(new FEM_Flexion_error_recorder(Vector3(4,0.5,0.5), Vector3(2.82376, -2.29429, 0.500275)));
        data_recorder->add(new FEM_Flexion_error_recorder(Vector3(4, 0.5, 0.5), Vector3(4, 0.5, 0.5) + Vector3(-0.213064, -1.22008, 0.)));
        e->addComponent(data_recorder);
    }
};
