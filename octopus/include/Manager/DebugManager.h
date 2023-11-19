#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Core/Engine.h"
#include "Mesh/Mesh.h"
#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_DisplayMode.h"
#include "Manager/Debug.h"
class DebugManager : public Behaviour
{
public:
    DebugManager() : _default_color(ColorBase::Grey(0.8)), _pause(false)
    {
        _graphic = new GL_Graphic();
        _graphic->set_multi_color(true);

        _display_mode = new GL_DisplayMesh();
        _display_mode->point() = false;
        _display_mode->surface() = false;

        _mesh = new Mesh();
        _mesh->set_dynamic_geometry(true);
        _mesh->set_dynamic_topology(true);

        Debug::Instance().set_mesh(_mesh);
        Debug::Instance().set_graphic(_graphic);
    }

    virtual void init() override { 
        Entity* root = Engine::GetEntity(0);
        root->addBehaviour(_mesh);
        root->addComponent(_graphic);
        root->addComponent(_display_mode);
    }

    virtual void update() override { 
        if (!_pause) {
            Debug::Axis(Unit3D::up() * scalar(0.005f), scalar(1.f));
            Debug::Instance().SetColor(ColorBase::Red());
            Debug::Cube(Vector3(0.05f), scalar(0.1f));
            Debug::Instance().SetColor(_default_color);
            Debug::UnitGrid(5);
        }
    }

    virtual void late_update() {
        if (!_pause) clear();
    }

    virtual void clear() {
        Debug::Instance().clear();
    }

    void play() {
        clear();
        _pause = false;
    }

    void pause() {
        _pause = true;
    }

protected:
    bool _pause;
    Color _default_color;
    GL_Graphic* _graphic;
    GL_DisplayMesh* _display_mode;
    Mesh* _mesh;
};