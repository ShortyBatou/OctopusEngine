#pragma once
#include "Core/Pattern.h"
#include "Rendering/gl_base.h"
#include <GLFW/glfw3.h>

struct AppInfo : public Singleton<AppInfo>
{
protected:
    friend Singleton<AppInfo>;
    AppInfo() : _exit(false), _height(0), _width(0)  { }

public:
    static bool Running()
    {
        return !Instance()._exit || !glfwWindowShouldClose(Instance()._window);
    }
    static void Window_sizes(unsigned int& w, unsigned int& h)
    {
        w = Instance()._width;
        h = Instance()._height;
    }
    static GLFWwindow* Window() { return Instance()._window; }
    static void Exit() { Instance()._exit = true; }
    static std::string PathToAssets() { return Instance()._path_to_assets; }
    void set_window_size(unsigned int w, unsigned int h)
    {
        _width  = w;
        _height = h;
    }
    void set_window(GLFWwindow* window) { _window = window; }
    void set_scene(unsigned int scene) {
        _scene = scene;
    }

    protected :
    unsigned int _width, _height;
    GLFWwindow* _window = nullptr;
    static const std::string _path_to_assets; 
    bool _exit;
    unsigned int _scene;
};

const std::string AppInfo::_path_to_assets = "assets/";