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
#include "Scene/BaseScene.h"
#include "Scene/XPBD_FEM_Torsion_Scene.h"

#include "UI/AppInfo.h"
#include "UI/UI_Editor.h"
#include "UI/UI_Component.h"
#include "Manager/Debug.h"
class Application : public Behaviour
{
public: 
    Application(unsigned int width = 1600, unsigned int height = 900) : _editor(nullptr)
    {
        auto& info = AppInfo::Instance(); // init app info
        info.set_window_size(width, height);

        Engine::Instance();
        SceneManager::Instance();
        SceneManager::Add(new BaseScene());
        SceneManager::Add(new XPBD_FEM_Torsion_Scene());
        SceneManager::SetScene(0);
    }

    ~Application() {
        delete _editor;
        Engine::Instance().clear();  // clear engine
        Engine::Delete();
        AppInfo::Delete();
        SceneManager::Delete();
        glfwTerminate();
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

    virtual void init() override { 
        init_glfw();
        init_imgui();
        Engine::Instance();  // init engine
        _editor = new UI_Editor();
        build_editor();
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
            { 
                glViewport(0, 0, width, height); 
                AppInfo::Instance().set_window_size(width, height);
            }
        );
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

    virtual void build_editor() {
        
        _editor->add_core_ui(new UI_SceneManager());
        _editor->init();
    }

    virtual void update() override {
        if (SceneManager::Instance().need_to_load()) {
            _editor->clear();
            SceneManager::Instance().load_scene(_editor);
            build_editor();
        }

        // init hud
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        Engine::Instance().update();
        _editor->draw();
        
        Engine::Instance().late_update();

        ImGui::ShowDemoWindow();

        DebugUI::Instance().draw();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(AppInfo::Window());  // glFibish();
        glfwPollEvents();
    }

protected:
    UI_Editor* _editor;
};
