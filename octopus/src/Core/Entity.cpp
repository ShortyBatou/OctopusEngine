#pragma once
#include <vector>
#include <string>
#include "Core/Entity.h"

Entity::Entity(const int id) : _id(id) {
    _name = "Entity_" + std::to_string(id);
}

Entity::Entity(const std::string &name, const int id) : _id(id), _name(name) {
}

void Entity::init() {
    // component can add new composents
    for(int i = 0; i < _components.size(); ++i) {
        _components[i]->init();
    }
}

void Entity::late_init() {
    for(int i = 0; i < _components.size(); ++i) {
        _components[i]->late_init();
    }
}

void Entity::update() {
    for(auto* _component : _components) {
        if (_component->active()) _component->update();
    }
}

void Entity::late_update() {
    for(auto* _component : _components) {
        if (_component->active()) _component->late_update();
    }
}

void Entity::add_behaviour(Behaviour *behaviour) {
    _components.push_back(behaviour);
}

void Entity::add_component(Component *component) {
    component->_entity = this;
    _components.push_back(component);
}

void Entity::remove_component(Component *component) {
    _components.erase(std::remove(_components.begin(), _components.end(), component), _components.end());
    delete component;
}

Entity::~Entity() {
    for (const auto* _component: _components) {
        delete _component;
    }
    _components.clear();
}
