#pragma once
#include "Core/Engine.h"
#include "Manager/DebugManager.h"
#include "Manager/Debug.h"

DebugManager::DebugManager(bool default_draw) : _default_color(ColorBase::Grey(0.8f)), _pause(false),
                                                _default_draw(default_draw) {
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

void DebugManager::init() {
    Entity *root = Engine::GetEntity(0);
    root->add_behaviour(_mesh);
    root->add_component(_graphic);
    root->add_component(_display_mode);
}

void DebugManager::update() {
    if (!_pause && _default_draw) {
        Debug::Axis(Unit3D::up() * 0.005f, 1.f);
        Debug::SetColor(ColorBase::Red());
        Debug::Cube(Vector3(0.05f), scalar(0.1f));
        Debug::SetColor(_default_color);
        Debug::UnitGrid(5);
    }
}

void DebugManager::late_update() {
    if (!_pause) clear();
}

void DebugManager::clear() {
    Debug::Instance().clear();
}

void DebugManager::play() {
    clear();
    _pause = false;
}
