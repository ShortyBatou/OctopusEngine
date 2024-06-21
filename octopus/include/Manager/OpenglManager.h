#pragma once

#include "Core/Pattern.h"
#include <Core/Engine.h>
#include "Tools/Color.h"

#include "Rendering/Camera.h"
#include "Rendering/Renderer.h"

#include "UI/AppInfo.h"

#include <map>
#include <iostream>

class OpenGLManager : public Behaviour
{
public: 
	OpenGLManager(const Color& background = ColorBase::Black())
        : _background(background)
    { 
		
	}

    virtual void init() override
    { 
        std::cout << "Init OpenglManager" << std::endl;
        for (Entity* e : Engine::GetEntities())
        {
            std::cout << "Search Renderers in " << e->name() << std::endl;
            std::vector<Renderer*> renderers = e->get_components<Renderer>();
            if (renderers.size() > 0)
            {
                _renderers.insert(_renderers.begin(), renderers.begin(), renderers.end());
                std::cout << "Renderers Found = " << renderers.size() << std::endl;
            }
        }
    }
    
    virtual void late_update() override
    { 
        // Render
        glClearColor(_background.r, _background.g, _background.b, _background.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
        glEnable(GL_DEPTH_TEST);  
        glDepthFunc(GL_LESS);

        for (Renderer* renderer : _renderers)
        {
            if (renderer->active())
                renderer->draw();
        }

        for (Renderer* renderer : _renderers)
        {
            if (renderer->active())
                renderer->after_draw();
        }

    }

    virtual ~OpenGLManager() {  }

    Color& background() {return _background;}

protected:
    std::vector<Renderer*> _renderers;
    Color _background;
};