#pragma once
#include "Scene.h"
#include "Manager/TimeManager.h"
#include "Manager/OpenglManager.h"
#include "Manager/CameraManager.h"
#include "Manager/InputManager.h"
#include "Manager/DebugManager.h"
#include "Manager/UI_Manager.h"

#include "Mesh/Mesh.h"
#include "Mesh/Generator/PrimitiveGenerator.h"
#include "Mesh/Generator/BeamGenerator.h"

#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_GraphicElement.h"
#include "Rendering/GL_DisplayMode.h"
#include "Rendering/GL_GraphicSurface.h"

#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"

struct BaseScene : public Scene
{
    virtual char* name() override { return "Basic Scene"; }

    virtual void init() override { 
        
    }

    virtual void build_root(Entity* root) override
    {
        root->addBehaviour(new TimeManager(1. / 60.));
        root->addBehaviour(new InputManager());
        root->addBehaviour(new CameraManager());
        root->addBehaviour(new DebugManager());
        root->addBehaviour(new UI_Manager());
        root->addBehaviour(new OpenGLManager(Color(0.9,0.9,0.9,1.)));
    }

    // build scene's entities
    virtual void build_entities() override
    { 
        Vector3I cells(16, 4, 4);
        Vector3 sizes(4, 1, 1);
        Entity* e    = Engine::CreateEnity();
        TetraBeamGenerator tm(cells, sizes);
        Mesh* mesh = tm.build();  // generate mesh

        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        e->addComponent(new XPBD_FEM_Dynamic());
        e->addComponent(new GL_GraphicSurface(Color(0.7, 0.3, 0.3, 1.)));
        e->addComponent(new GL_DisplayMesh());


        e = Engine::CreateEnity();
        tm.setTransform(glm::translate(Matrix::Identity4x4(), Vector3(0., 0., 1.5)));
        mesh = tm.build();  // generate mesh
        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        e->addComponent(new FEM_Dynamic());
        e->addComponent(new GL_GraphicSurface(Color(0.3, 0.3, 0.7, 1.)));
        e->addComponent(new GL_DisplayMesh());
    }
};