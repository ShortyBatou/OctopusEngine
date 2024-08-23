#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Mesh/Mesh.h"
#include "Rendering/GL_DisplayMode.h"

class DebugManager : public Behaviour {
public:
    explicit DebugManager(bool default_draw = true);

    void init() override;

    void update() override;

    void late_update() override;

    void clear();

    void play();

    void pause() { _pause = true; }

    bool &default_draw() { return _default_draw; }

protected:
    bool _default_draw;
    bool _pause;
    Color _default_color{};
    GL_Graphic *_graphic;
    GL_DisplayMesh *_display_mode;
    Mesh *_mesh;
};
