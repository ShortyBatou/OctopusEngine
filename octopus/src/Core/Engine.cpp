#pragma once
#include <string>
#include "Core/Engine.h"

void Engine::init() {
    for (auto &_entitie: _entities)
        _entitie->init();
}

void Engine::late_init() {
    for (auto &_entitie: _entities)
        _entitie->late_init();
}

void Engine::update() {
    for (int i = 1; i < _entities.size(); ++i)
        if (_entities[i]->active()) _entities[i]->update();

    _entities[0]->update(); // always update root at the end
}

void Engine::late_update() {
    for (int i = 1; i < _entities.size(); ++i)
        if (_entities[i]->active()) _entities[i]->late_update();

    _entities[0]->late_update(); // always update root at the end
}

Entity *Engine::CreateEnity() {
    auto &engine = Engine::Instance();
    Entity *e = new Entity(int(engine._entities.size()));
    engine._entities.push_back(e);
    return e;
}

Entity *Engine::CreateEnity(const std::string &name) {
    auto &engine = Engine::Instance();
    Entity *e = new Entity(name, int(engine._entities.size()));
    engine._entities.push_back(e);
    return e;
}


Entity *Engine::GetEntity(const std::string &name) {
    auto &engine = Engine::Instance();
    for (Entity *e: engine._entities)
        if (e->name() == name) return e;
    return nullptr;
}

Entity *Engine::GetEntity(int id) {
    auto &engine = Engine::Instance();
    for (auto &_entitie: engine._entities)
        if (id == _entitie->id())
            return _entitie;
    return nullptr;
}

std::vector<Entity *> &Engine::GetEntities() {
    auto &engine = Engine::Instance();
    return engine._entities;
}

int Engine::Count() {
    auto &engine = Engine::Instance();
    return int(engine._entities.size());
}

void Engine::clear() {
    for (auto &_entitie: _entities) {
        delete _entitie;
    }
    _entities.clear();
}

Engine::~Engine() { clear(); }
