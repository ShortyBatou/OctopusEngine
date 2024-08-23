#pragma once
#include "Core/Pattern.h"
#include "Tools/Color.h"
#include "Rendering/Renderer.h"
#include <vector>

class OpenGLManager : public Behaviour {
public:
    explicit OpenGLManager(const Color &background = ColorBase::Black())
        : _background(background) {
    }

    void init() override;

    void late_update() override;

    Color &background() { return _background; }

protected:
    std::vector<Renderer *> _renderers;
    Color _background;
};
