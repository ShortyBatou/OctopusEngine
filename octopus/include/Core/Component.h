#pragma once
#include "Pattern.h"

struct Entity;

struct Component : Behaviour {
    friend Entity;

    Component() : _entity(nullptr) { }

    inline Entity* entity() { return _entity; }

protected:
    Entity* _entity;
};