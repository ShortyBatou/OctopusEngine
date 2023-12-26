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
#include "Script/VTK/VTK_FEM.h"
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
        editor->add_component_ui(new UI_FEM_Saver());
    }

    virtual void build_root(Entity* root) override
    {
        root->addBehaviour(new TimeManager(1. / 60.));
        root->addBehaviour(new InputManager());
        root->addBehaviour(new CameraManager());
        root->addBehaviour(new DebugManager(true));
        root->addBehaviour(new OpenGLManager(Color(0.9,0.9,0.9,1.)));
    }

    // build scene's entities
    virtual void build_entities() override 
    {  
        Vector3 size(1, 1, 4);
        Vector3I cells(4, 4, 12);
        //build_xpbd_entity(Vector3(0., 0., 0), cells, size, Color(0.3, 0.8, 0.3, 1.), Tetra, true);
        //build_xpbd_entity(Vector3(1., 0., 0.), cells, size, Color(0.8, 0.3, 0.3, 1.), Hexa, false);
        //build_xpbd_entity(Vector3(0., 0., 0), cells, size, Color(0.3, 0.3, 0.8, 1.), Tetra10, false, true);

        //cells = Vector3I(1, 1, 1);
        build_xpbd_entity(Vector3(0., 0., 0.), cells, size, Color(0.8, 0.3, 0.3, 1.), Tetra, false);

        //build_xpbd_entity(Vector3(0., 0, 0.), cells, size, Color(0.3, 0.3, 0.8, 1.), Tetra, true);
        //build_xpbd_entity(Vector3(2., 0., 0.), cells, size, Color(0.3, 0.8, 0.3, 1.), Tetra10, false, true);
    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, Element element) {
        BeamMeshGenerator* generator;
        switch (element)
        {
            case Tetra: generator = new TetraBeamGenerator(cells, size); break;
            case Pyramid: generator = new PyramidBeamGenerator(cells, size); break;
            case Prysm: generator = new PrysmBeamGenerator(cells, size); break;
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
        
        //Mesh* mesh = get_beam_mesh(pos, cells, size, element);

        Msh_Loader loader(AppInfo::PathToAssets() + "/mesh/bar_1300.msh");
        loader.setTransform(glm::scale(Vector3(0.5)) * glm::translate(Matrix::Identity4x4(), pos + Vector3(0.,0.,4.) ));
        Mesh* mesh = loader.build();

        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());

        scalar density = 100;
        scalar young = 100000;
        scalar poisson = 0.49;
        Material material = Developed_Neohooke;
        unsigned int sub_it = 30;
        Vector3 dir = Unit3D::forward();
        unsigned int scenario_1 = 1;
        unsigned int scenario_2 = 1;

        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        if (fem) {
            e->addComponent(new FEM_Dynamic(density, young, poisson, material, 300));
        }
        else {
            e->addComponent(new XPBD_FEM_Dynamic(density, young, poisson, material, 1, sub_it, GaussSeidel, pbd_v1));
        }

        e->addComponent(new VTK_FEM(
            std::string(element_name(element)) 
            + "_quality" 
            + "_subit" + std::to_string(int(sub_it)) +
            + "_p" + std::to_string(int(density))
            + "_v" + std::to_string(int(young))
            + "_E" + std::to_string(int(poisson * 100)) + "_Developed_NeoHooke_H2"
        ));

        //e->addComponent(new VTK_FEM("base"));

        e->addComponent(new Constraint_Rigid_Controller(dir * scalar(0.01), -dir, scenario_1));
        e->addComponent(new Constraint_Rigid_Controller(pos - dir * scalar(0.01) + size, dir, scenario_2));
        GL_Graphic* graphic;
        if (element == Tetra10)
            graphic = new GL_GraphicHighOrder(0, color);
            //graphic = new GL_GraphicElement(1.);
        else
            graphic = new GL_GraphicSurface(color);

        graphic->normals() = false;

        e->addComponent(graphic);
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->point() = false;
        display->normal() = false;
        e->addComponent(display);
    }
};