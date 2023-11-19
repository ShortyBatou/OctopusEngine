#pragma once
#include "Core/Engine.h"
#include "Core/Entity.h"
#include <UI/UI_Editor.h>

struct Scene
{
    virtual char* name() = 0;
    
    virtual void init() { }

    // gives the editor object that it can be modified for this scene
    virtual void build_editor(UI_Editor* editor) { }

    // build the root entity that will contains all global behaviours and managers
    virtual void build_root(Entity* root) = 0;
    
    // build scene's entities
    virtual void build_entities() = 0;
};