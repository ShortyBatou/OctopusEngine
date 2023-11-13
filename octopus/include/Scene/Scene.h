#pragma once
#include "Core/Engine.h"
#include "Core/Entity.h"

struct Scene
{
    virtual char* name() = 0;
    
    virtual void init() { }

    // build the root entity that will contains all global behaviours and managers
    virtual void build_root(Entity* root) = 0;
    
    // build scene's entities
    virtual void build_entities() = 0;
};