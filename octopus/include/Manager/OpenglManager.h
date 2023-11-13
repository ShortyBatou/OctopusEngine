#pragma once
#include <map>

#include "Core/Pattern.h"
#include "Tools/Color.h"
#include "HUD/AppInfo.h"
#include "Rendering/Camera.h"
#include "Rendering/GL_DisplayMode.h"
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
            std::cout << "Search Display Mode in " << e->name() << std::endl;
            GL_DisplayMode* display_mode = e->getComponent<GL_DisplayMode>();
            if (display_mode != nullptr)
            {
                renderers.push_back(display_mode);
                std::cout << "display_mode found" << std::endl;
            }
        }
    }
    
    virtual void update() override
    { 
        // Render
        glClearColor(_background.r, _background.g, _background.b, _background.a);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
        glEnable(GL_DEPTH_TEST);  
        glDepthFunc(GL_LESS);

        for (GL_DisplayMode* renderer : renderers)
        {
            if (renderer->active())
                renderer->draw();
        }

        glfwSwapBuffers(AppInfo::Window());  // glFibish();
        glfwPollEvents();
    }

    virtual ~OpenGLManager() {  }

protected:
    std::vector<GL_DisplayMode*> renderers;
    Color _background;
};