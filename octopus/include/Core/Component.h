#pragma once
#include "Pattern.h"

struct Entity;

struct Component : Behaviour {
    friend Entity; 

    explicit Component(Entity* entity = nullptr) : _entity(entity) { }

    [[nodiscard]] Entity* entity() const { return _entity; }

protected:
    Entity* _entity;
};