#pragma once
#include <string>
#include "Core/Engine.h"

void Engine::init() {
    for (auto *_entitie: _entities)
        _entitie->init();
}

void Engine::late_init() {
    for (auto *_entitie: _entities)
        _entitie->late_init();
}

void Engine::update() {
    for (int i = 1; i < _entities.size(); ++i)
        if (_entities[i]->active())
            _entities[i]->update();

    _entities[0]->update(); // always update root at the end
}

void Engine::late_update() {
    for (int i = 1; i < _entities.size(); ++i)
        if (_entities[i]->active())
            _entities[i]->late_update();

    _entities[0]->late_update(); // always update root at the end
}

Entity *Engine::CreateEnity() {
    auto &engine = Instance();
    Entity *e = new Entity(static_cast<int>(engine._entities.size()));
    engine._entities.push_back(e);
    return e;
}

Entity *Engine::CreateEnity(const std::string &name) {
    auto &engine = Instance();
    Entity *e = new Entity(name, static_cast<int>(engine._entities.size()));
    engine._entities.push_back(e);
    return e;
}


Entity *Engine::GetEntity(const std::string &name) {
    const auto &engine = Instance();
    for (Entity *e: engine._entities)
        if (e->name() == name) return e;
    return nullptr;
}

Entity *Engine::GetEntity(const int id) {
    const auto &engine = Instance();
    for (auto *e: engine._entities)
        if (id == e->id()) return e;
    return nullptr;
}

std::vector<Entity *> &Engine::GetEntities() {
    auto &engine = Instance();
    return engine._entities;
}

int Engine::Count() {
    const auto &engine = Instance();
    return static_cast<int>(engine._entities.size());
}

void Engine::clear() {
    for (const auto *e: _entities) {
        delete e;
    }
    _entities.clear();
}

Engine::~Engine() { clear(); }
