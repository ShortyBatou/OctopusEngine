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

struct XPBD_FEM_Torsion_Scene : public Scene
{
    virtual char* name() override { return "Torsion XPBD FEM"; }

    virtual void init() override 
    { 
    
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
        root->addBehaviour(new CameraManager(Unit3D::up() * 8.f  + Unit3D::right() * 0.001f));
        root->addBehaviour(new DebugManager(false));
        root->addBehaviour(new OpenGLManager(Color(1., 1., 1., 1.)));
    }

    // build scene's entities
    virtual void build_entities() override
    {
        Vector3I cells(8, 4, 4);
        Vector3 size(2, 1, 1);
        build_xpbd_entity(Vector3(0, 0, 0), cells, size, Color(0.3, 0.3, 0.8, 1.), Tetra10);
    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3I& size, Element element) {
        BeamMeshGenerator* generator;
        switch (element)
        {
            case Tetra: generator = new TetraBeamGenerator(cells, size); break;
            case Pyramid: generator = new PyramidBeamGenerator(cells, size); break;
            case Prysm: generator = new PrysmBeamGenerator(cells, size); break;
            case Hexa: generator = new HexaBeamGenerator(cells, size); break;
            case Tetra10: generator = new TetraBeamGenerator(cells, size); break; 
            default: break;
        }
        generator->setTransform(glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = generator->build();
        delete generator;
        return mesh;
    }

    void build_xpbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element) {
        Entity* e = Engine::CreateEnity();

        Mesh* mesh = get_beam_mesh(pos, cells, size, element);


        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());

        mesh->set_dynamic_geometry(true);
        e->addBehaviour(mesh);
        e->addComponent(new XPBD_FEM_Dynamic(100, 100000, 0.499, Neo_Hooke, 1, 40, GaussSeidel));
        e->addComponent(new Constraint_Rigid_Controller(pos + Unit3D::right() * scalar(0.01), -Unit3D::right()));
        e->addComponent(new Constraint_Rigid_Controller(pos - Unit3D::right() * scalar(0.01) + size, Unit3D::right()));
        GL_Graphic* graphic;
        if (element == Tetra10)
            graphic = new GL_GraphicHighOrder(3, color);
        else
            graphic = new GL_GraphicSurface(color);

        graphic->normals() = true;

        e->addComponent(graphic);
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->point() = false;
        display->normal() = false;
        e->addComponent(display);
    }
};