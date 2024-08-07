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
        editor->add_manager_ui(new UI_DisplaySettings());
        editor->add_manager_ui(new UI_Camera());
        editor->add_component_ui(new UI_Mesh_Display());
    }

    virtual void build_root(Entity* root) override
    {
        root->add_behaviour(new TimeManager(1.f / 60.f));
        root->add_behaviour(new InputManager());
        root->add_behaviour(new CameraManager(Unit3D::up() * 8.f  + Unit3D::right() * 0.001f));
        root->add_behaviour(new DebugManager(false));
        root->add_behaviour(new OpenGLManager(Color(1., 1., 1., 1.)));
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
            case Prism: generator = new PrismBeamGenerator(cells, size); break;
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
        e->add_behaviour(mesh);
        e->add_component(new XPBD_FEM_Dynamic(100, 100000, 0.49f, NeoHooke, 1, 50, GaussSeidel));
        e->add_component(new Constraint_Rigid_Controller(pos + Unit3D::right() * 0.01f, -Unit3D::right()));
        e->add_component(new Constraint_Rigid_Controller(pos - Unit3D::right() * 0.01f + size, Unit3D::right()));
        GL_Graphic* graphic;
        //if (element == Tetra10)
        //    graphic = new GL_GraphicHighOrder(3, color);
        //else
            graphic = new GL_GraphicSurface(color);


        e->add_component(graphic);
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->point() = false;
        e->add_component(display);
    }
};