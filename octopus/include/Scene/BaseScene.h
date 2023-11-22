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

#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_GraphicElement.h"
#include "Rendering/GL_DisplayMode.h"
#include "Rendering/GL_GraphicSurface.h"
#include "Rendering/GL_GraphicHighOrder.h"

#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"

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
    }

    virtual void build_root(Entity* root) override
    {
        root->addBehaviour(new TimeManager(1. / 60.));
        root->addBehaviour(new InputManager());
        root->addBehaviour(new CameraManager());
        root->addBehaviour(new DebugManager());
        root->addBehaviour(new OpenGLManager(Color(0.9,0.9,0.9,1.)));
    }

    // build scene's entities
    virtual void build_entities() override
    { 
        Vector3I cells(4, 1, 1);
        Vector3 sizes(4, 1, 1);
        Entity* e    = Engine::CreateEnity();
        TetraBeamGenerator tm(cells, sizes);
        Mesh* mesh = tm.build();  // generate mesh
        tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        e->addComponent(new XPBD_FEM_Dynamic());
        e->addComponent(new GL_GraphicHighOrder(3, Color(0.7, 0.3, 0.3, 1.)));
        //e->addComponent(new GL_GraphicElement());
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->normal() = true;
        e->addComponent(display);

        cells = Vector3(8, 2, 2);
        TetraBeamGenerator tm2(cells, sizes);
        e = Engine::CreateEnity();
        tm2.setTransform(glm::translate(Matrix::Identity4x4(), Vector3(0, 0, 2)));
        mesh = tm2.build();  // generate mesh
        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        e->addComponent(new XPBD_FEM_Dynamic());
        e->addComponent(new GL_GraphicSurface(Color(0.3, 0.3, 0.7, 1.)));
        display = new GL_DisplayMesh();
        display->normal() = true;
        e->addComponent(display);
    }
};