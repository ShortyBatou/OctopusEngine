#pragma once
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "Rendering/gl_base.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "Core/Engine.h"
#include "Core/Pattern.h"
#include "Core/Base.h"
#include "Scene/SceneManager.h"
#include "UI/AppInfo.h"

class Application : public Behaviour
{
public: 
    Application(unsigned int width = 1280, unsigned int height = 720)
        : _scene_manager(new SceneManager())
    {
        auto& info = AppInfo::Instance(); // init app info
        info.set_window_size(width, height);
    }

    ~Application() { 
        Engine::Instance().clear();  // clear engine
        Engine::Delete();
        AppInfo::Delete();
        glfwTerminate();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        delete _scene_manager;
    }

    virtual void init() override { 
        init_glfw();
        init_imgui();
        Engine::Instance();  // init engine
        _scene_manager->build(); // build the default scene
    }

    /// Create Opengl Context
    void init_glfw() {
        unsigned int w, h; AppInfo::Window_sizes(w, h);

        // GLFW
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

        // WINDOW
        AppInfo::Instance().set_window( glfwCreateWindow(w, h, "Octopus", nullptr, nullptr));
        if (AppInfo::Window() == nullptr)
        {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            AppInfo::Exit();
            return;
        }
        glfwMakeContextCurrent(AppInfo::Window());

        // GLEW
        if (glewInit() != GLEW_OK) std::cout << "Error in glew init" << '\n';

        std::cout << glGetString(GL_VERSION) << '\n';

        // CALLBACKS
        // KEYS
        // a utiliser avec l'input manager
        glfwSetKeyCallback(
            AppInfo::Window(),
            [](GLFWwindow* window, int key, int scancode, int action, int mods)
            {
                if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
                {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
            });

        // RESIZE
        glfwSetFramebufferSizeCallback(
            AppInfo::Window(), [](GLFWwindow* window, int width, int height)
            { glViewport(0, 0, width, height); });
    }

    /// <summary>
    /// Create imgui context
    /// </summary>
    void init_imgui() {
        const char* glsl_version = "#version 430";
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        // Style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(AppInfo::Window(), true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    virtual void update() override {
        Engine::Instance().update();
        Engine::Instance().late_update();
    }

protected:
   
    SceneManager* _scene_manager;
};
