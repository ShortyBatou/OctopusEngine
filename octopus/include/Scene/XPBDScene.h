#pragma once
#include "Scene.h"
#include "Manager/TimeManager.h"
#include "Manager/OpenglManager.h"
#include "Manager/CameraManager.h"
#include "Manager/InputManager.h"
#include "Manager/DebugManager.h"
#include "Manager/Dynamic.h"

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
#include "Script/Display/DataDisplay.h"

struct XPBDScene : public Scene
{
    virtual char* name() override { return "Torsion XPBD FEM"; }

    virtual void init() override 
    { 
    
    }

    virtual void build_editor(UI_Editor* editor) {
        editor->add_manager_ui(new UI_Time());
        editor->add_manager_ui(new UI_Dynamic());
        editor->add_manager_ui(new UI_DisplaySettings());
        editor->add_manager_ui(new UI_Camera());
        editor->add_component_ui(new UI_Mesh_Display());
        editor->add_component_ui(new UI_Data_Recorder());
        editor->add_component_ui(new UI_Data_Displayer());
        editor->add_component_ui(new UI_Graphic_Saver());
        editor->add_component_ui(new UI_PBD_Dynamic());
        editor->add_component_ui(new UI_Constraint_Rigid_Controller());
    }

    virtual void build_root(Entity* root) override
    {
        root->add_behaviour(new TimeManager(1.f / 120.f));
        root->add_behaviour(new DynamicManager(Vector3(0.,-9.81*0.f,0.)));
        root->add_behaviour(new InputManager());
        root->add_behaviour(new CameraManager());
        root->add_behaviour(new DebugManager(true));
        root->add_behaviour(new OpenGLManager(Color(0.9f,0.9f,0.9f,1.f)));
    }

    // build scene's entities
    virtual void build_entities() override
    {
        SimulationArgs args;
        args.density = 1000;
        args.material = Stable_NeoHooke;
        args.poisson = 0.49f;
        args.young = 1e6f;
        args.damping = 1.;
        args.iteration = 1;
        args.sub_iteration = 25;
        args.scenario_1 = 0;
        args.scenario_2 = -1;
        args.dir = Unit3D::up();

        Vector3 size(1, 1, 1);
        Vector3I cells;
        cells = Vector3I(6, 6, 6);
        build_xpbd_fem_entity(Vector3(0, 0, 0.), cells, size, Color(0.f, 0.f, 0.f, 1.f), Tetra10, args);
        //cells = Vector3I(12, 4, 4);
        //build_xpbd_fem_entity(Vector3(0, 0, -0.8), cells, size, Color(0.f, 0.f, 0.f, 1.f), Hexa, args);
        //cells = Vector3I(6, 2, 2);
        //build_xpbd_fem_entity(Vector3(0, 0, 0.8), cells, size, Color(0.f, 0.f, 0.f, 1.f), Tetra10, args);
        //cells = Vector3I(4, 2, 2);
        //build_xpbd_fem_entity(Vector3(0, 0, 2.4), cells, size, Color(0.f, 0.f, 0.f, 1.f), Tetra20, args);
        //build_fem_entity(Vector3(0, 0, 0), cells, size, Color(0.0f, 0.0f, 0.0f, 1.f), Hexa, args);
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
            default: break;
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

    GL_Graphic* build_graphic(const Color& color, Element element) {
        // Mesh converter simulation to rendering (how it will be displayed)
        GL_Graphic* graphic;

        if (element == Tetra10 || element == Tetra20)
            graphic = new GL_GraphicHighOrder(3, color);
        else
            graphic = new GL_GraphicSurface(color);

        return graphic;
    }

    GL_DisplayMesh* build_display() {
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->surface() = true;
        display->wireframe() = true;
        display->point() = false;
        return display;
    }

    void add_constraint(Entity* e, const Vector3& pos, const Vector3& size, SimulationArgs& args) {
        // constraint for Particle system
        if (args.scenario_1 != -1) {
            auto rd_constraint_1 = new Constraint_Rigid_Controller(Unit3D::Zero() + args.dir*0.01f, -args.dir, args.scenario_1);
            e->add_component(rd_constraint_1);
            rd_constraint_1->_rot_speed = 90;
            rd_constraint_1->_move_speed = 1.;
        }

        if (args.scenario_2 != -1) {
            auto rd_constraint_2 = new Constraint_Rigid_Controller(pos + size, args.dir, args.scenario_2);
            rd_constraint_2->_rot_speed = -90;
            rd_constraint_2->_move_speed = 1.;
            rd_constraint_2->_event_rate = 1.5;
            rd_constraint_2->_smooth_iterations = 10;
            e->add_component(rd_constraint_2);
        }
    }

    Mesh* build_vtk_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, std::string file) {
        VTK_Loader loader(AppInfo::PathToAssets() + file);
        loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = loader.build();
        mesh->set_dynamic_geometry(true);
        return mesh;
    }

    DataRecorder* build_data_recorder(const Vector3I& cells, const Vector3& size, Element element) {
        std::string file_name = std::string(element_name(element)) + "_" + std::to_string(cells.x) + "_" + std::to_string(cells.y) + "_" + std::to_string(cells.z)
            + "_" + std::to_string(int(size.x)) + "x" + std::to_string(int(size.y)) + "x" + std::to_string(int(size.z));
        DataRecorder* data_recorder = new DataRecorder(file_name, true);
        data_recorder->add(new TimeRecorder());
        data_recorder->add(new MeshRecorder());
        data_recorder->add(new FEM_Dynamic_Recorder());
        data_recorder->add(new FEM_VTK_Recorder(file_name));
        data_recorder->add(new Graphic_VTK_Recorder(file_name));
        //data_recorder->add(new FEM_Flexion_error_recorder(Vector3(4,0.5,0.5), Vector3(2.82376, -2.29429, 0.500275)));
        //data_recorder->add(new FEM_Flexion_error_recorder(Vector3(4, 0.5, 0.5), Vector3(4, 0.5, 0.5) + Vector3(-0.213064, -1.22008, 0.)));
        return data_recorder;
    }

    void build_fem_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element, SimulationArgs& args) {
        Entity* e = Engine::CreateEnity();
        e->add_behaviour(build_beam_mesh(pos, cells, size, element));
        e->add_component(new FEM_Dynamic(args.density, args.young, args.poisson, args.material, args.sub_iteration));
        add_constraint(e, pos, size, args);
        e->add_component(new FEM_DataDisplay(FEM_DataDisplay::Type::Displacement, ColorMap::Viridis));
        e->add_component(build_graphic(color, element));
        e->add_component(build_display());
        e->add_component(build_data_recorder(cells, size, element));
    }

    void build_xpbd_fem_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element, SimulationArgs& args) {
        Entity* e = Engine::CreateEnity();
        e->add_behaviour(build_beam_mesh(pos, cells, size, element));
        //e->add_behaviour(build_vtk_mesh(pos, cells, size, color, "mesh/vtk/bunny_P3.vtk"));
        e->add_component(new XPBD_FEM_Dynamic(args.density, args.young, args.poisson, args.material, args.iteration, args.sub_iteration, args.damping));
        add_constraint(e, pos, size, args);
        e->add_component(new FEM_DataDisplay(FEM_DataDisplay::Type::Volume_Diff, ColorMap::Viridis));
        e->add_component(build_graphic(color, element));
        e->add_component(build_display());
        e->add_component(build_data_recorder(cells, size, element));
    }
};