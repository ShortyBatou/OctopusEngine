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
        Vector3 size(4, 1, 1);
        Vector3I cells;

        cells = Vector3I(16,4,4);
        //build_xpbd_entity(Vector3(0, 0, 0), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra10, false, false);
        build_xpbd_entity(Vector3(0, 0, 0), cells, size, Color(0.8f, 0.3f, 0.8f, 1.f), Hexa, false, false);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.3, 0.8, 0.3, 1.), Tetra20, false, false);
        //build_xpbd_entity(Vector3(0, 0, 2), cells, size, Color(0.3, 0.3, 0.8, 1.), Tetra20, false, false);
        //cells = Vector3I(8, 3, 3);
        //cells = Vector3I(6, 2, 2);
        //build_xpbd_entity(Vector3(0, 0, 2), cells, size, Color(0.8, 0.3, 0.3, 1.), Tetra, false, true);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra10, true, false);
    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, Element element) {
        BeamMeshGenerator* generator;
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

    void build_xpbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element, bool pbd_v1 = false, bool fem = false) {
        Entity* e = Engine::CreateEnity();

        // Mesh generation or loading
        Mesh* mesh;

        //Msh_Loader loader(AppInfo::PathToAssets() + "mesh/msh/beam_tetra_8x2x2.msh");
        //mesh = loader.build();

        //if (element == Tetra || element == Tetra10 || element == Tetra20) {
        //    std::string path = "";
        //    if (cells.x == 18) {
        //        path = "mesh/vtk/beam-s-3-1-1-n-18-6-6-tetra.vtk";
        //    }
        //    else if (cells.x == 9) {
        //        path = "mesh/vtk/beam-s-3-1-1-n-9-3-3-tetra.vtk";
        //    }
        //    else if (cells.x == 6) {
        //        path = "mesh/vtk/beam-s-3-1-1-n-6-2-2-tetra.vtk";
        //    }
        //    VTK_Loader loader(AppInfo::PathToAssets() + path);
        //    loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos + Vector3(0., 0., 0.)));
        //    mesh = loader.build();
        //}
        //else {
        //    mesh = get_beam_mesh(pos, cells, size, element);
        //}

        mesh = get_beam_mesh(pos, cells, size, element);
        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
  
        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        
        // simulation FEM or PBD
        scalar density = 1000;
        scalar young = 1e6;
        scalar poisson = 0.35f;
        Material material = Developed_Neohooke;
        int sub_it = 50;
        scalar global_damping = 3.;
        Vector3 dir = Unit3D::right();
        int scenario_1 = 0;
        int scenario_2 = 0;

        if (fem) {
            e->addComponent(new FEM_Dynamic(density, young, poisson, material, 300));
        }
        else {
            e->addComponent(new XPBD_FEM_Dynamic(density, young, poisson, material, 1, sub_it, global_damping, GaussSeidel));
        }

        // constraint for Particle system
        auto rd_constraint_1 = new Constraint_Rigid_Controller(Unit3D::Zero(), -dir, scenario_1);
        e->addComponent(rd_constraint_1);
        rd_constraint_1->_rot_speed = 90;
        rd_constraint_1->_move_speed = 1;

        //auto rd_constraint_2 = new Constraint_Rigid_Controller(pos + size, dir, scenario_2);
        //rd_constraint_2->_rot_speed = 180;
        //rd_constraint_2->_move_speed = 1;
        //e->addComponent(rd_constraint_2);

        //auto cf_c = new ConstantForce_Controller(Vector3(0.5, 0.5, 0.0), Vector3(1, 1, 1), Unit3D::right() * 5.f);
        //e->addComponent(cf_c);

        //e->addComponent(new Constraint_Rigid_Controller(dir * scalar(0.01), -Unit3D::right(), scenario_1));
        //e->addComponent(new Constraint_Rigid_Controller(pos - dir * scalar(0.01) + size, Unit3D::right(), scenario_2));

        //e->addComponent(new Constraint_Rigid_Controller(dir * scalar(0.01), -Unit3D::forward(), scenario_1));
        //e->addComponent(new Constraint_Rigid_Controller(pos - dir * scalar(0.01) + size, Unit3D::forward(), scenario_2));

        // Mesh converter simulation to rendering (how it will be displayed)
        GL_Graphic* graphic;

        //graphic = new GL_GraphicElement(0.7);
        if (element == Tetra10 || element == Tetra20)
            //graphic = new GL_GraphicElement(0.7);
            graphic = new GL_GraphicHighOrder(3, color);
        else
            graphic = new GL_GraphicSurface(color);

        graphic->normals() = false;
        e->addComponent(graphic);

        // Opengl Rendering
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->wireframe() = true;
        display->point() = false;
        display->normal() = false;
        e->addComponent(display);

        // save mesh in VTK format (Paraview)size

        std::string file_name = std::string(element_name(element)) + "_" + std::to_string(cells.x) + "_" + std::to_string(cells.y) + "_" + std::to_string(cells.z) 
            + "_" + std::to_string(int(size.x)) + "x" + std::to_string(int(size.y)) + "x" + std::to_string(int(size.z));
        DataRecorder* data_recorder = new DataRecorder(file_name);
        data_recorder->add(new TimeRecorder());
        data_recorder->add(new MeshRecorder());
        data_recorder->add(new XPBD_FEM_Dynamic_Recorder());
        data_recorder->add(new XPBD_FEM_VTK_Recorder(file_name));
        data_recorder->add(new Graphic_VTK_Recorder(file_name));
        //data_recorder->add(new FEM_Flexion_error_recorder(Vector3(4,0.5,0.5), Vector3(2.82376, -2.29429, 0.500275)));
        //data_recorder->add(new FEM_Torsion_error_recorder(180, size.x));
        e->addComponent(data_recorder);
    }
};
