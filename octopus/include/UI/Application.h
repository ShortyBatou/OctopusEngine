#pragma once

#include "Core/Pattern.h"
#include "UI/UI_Editor.h"

class Application : public Behaviour
{
public: 
    explicit Application(int width = 1600, int height = 900);

    ~Application() override;

    void init() override;

    /// Create Opengl Context
    void init_glfw();

    /// <summary>
    /// Create imgui context
    /// </summary>
    void init_imgui();

    virtual void build_editor();

    void update() override;

protected:
    UI_Editor* _editor;
};
