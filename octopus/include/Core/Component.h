#pragma once
#include "Pattern.h"

struct Entity;

struct Component : Behaviour {
    friend Entity; 

    Component(Entity* entity = nullptr) : _entity(entity) { }

    inline Entity* entity() { return _entity; }

protected:
    Entity* _entity;
};