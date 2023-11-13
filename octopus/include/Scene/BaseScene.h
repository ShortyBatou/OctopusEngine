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
        root->addBehaviour(new OpenGLManager(Color(0.7,0.7,0.95,1.)));
    }

    // build scene's entities
    virtual void build_entities() override
    { 
        Vector3I cells(6, 2, 2);
        Vector3 sizes(2, 1, 1);
        Entity* e    = Engine::CreateEnity();
        Vector3 pmin = Vector3(-0.5, 2, -0.5);
        Vector3 pmax  = Vector3(0.5, 2.5, 0.5);
        PyramidBeamGenerator tm2(cells, sizes);
        Mesh* mesh = tm2.build();  // generate mesh
        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        GL_GraphicSurface* graphic = new GL_GraphicSurface();
        e->addComponent(graphic);
        e->addComponent(new GL_DisplayMesh());
    }
};