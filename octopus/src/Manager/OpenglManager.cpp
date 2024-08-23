#pragma once
#include "Rendering/gl_base.h"
#include "Core/Engine.h"
#include "Rendering/Camera.h"
#include "Manager/OpenglManager.h"
#include <iostream>

void OpenGLManager::init() {
    std::cout << "Init OpenglManager" << std::endl;
    for (Entity *e: Engine::GetEntities()) {
        std::cout << "Search Renderers in " << e->name() << std::endl;
        std::vector<Renderer *> renderers = e->get_components<Renderer>();
        if (!renderers.empty()) {
            _renderers.insert(_renderers.begin(), renderers.begin(), renderers.end());
            std::cout << "Renderers Found = " << renderers.size() << std::endl;
        }
    }
}

void OpenGLManager::late_update() {
    // Render
    glClearColor(_background.r, _background.g, _background.b, _background.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    for (Renderer *renderer: _renderers) {
        if (renderer->active())
            renderer->draw();
    }

    for (Renderer *renderer: _renderers) {
        if (renderer->active())
            renderer->after_draw();
    }
}
