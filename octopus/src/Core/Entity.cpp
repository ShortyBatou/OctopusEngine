#pragma once
#include <vector>
#include <string>
#include "Core/Entity.h"

Entity::Entity(const int id) : _id(id) {
    _name = "Entity_" + std::to_string(id);
}

Entity::Entity(const std::string &name, int id) : _name(name), _id(id) {
}

void Entity::init() {
    for (auto &_component: _components)
        _component->init();
}

void Entity::late_init() {
    for (auto &_component: _components)
        _component->late_init();
}

void Entity::update() {
    for (auto &_component: _components)
        if (_component->active()) _component->update();
}

void Entity::late_update() {
    for (auto &_component: _components)
        if (_component->active()) _component->late_update();
}

template<class T>
T *Entity::get_component() {
    for (auto &_component: _components) {
        if (typeid(*_component) == typeid(T))
            return dynamic_cast<T *>(_component);

        T *_c = dynamic_cast<T *>(_component);
        if (_c != nullptr) return _c;
    }

    return nullptr;
}

template<class T>
std::vector<T *> Entity::get_components() {
    std::vector<T *> list;
    for (auto &_component: _components) {
        if (typeid(*_component) == typeid(T)) {
            list.push_back(static_cast<T *>(_component));
        } else {
            T *_c = dynamic_cast<T *>(_component);
            if (_c != nullptr)
                list.push_back(_c);
        }
    }
    return list;
}

template<class T>
T *Entity::get_component_by_id(int i) {
    if (i >= _components.size()) return nullptr;

    return static_cast<T>(_components[i]);
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
    for (auto &_component: _components) {
        delete _component;
    }
    _components.clear();
}
