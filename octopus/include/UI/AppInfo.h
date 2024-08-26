#pragma once
#include "Core/Pattern.h"
#include "Rendering/gl_base.h"
#include <GLFW/glfw3.h>
struct GLFWwindow;

struct AppInfo : Singleton<AppInfo>
{
protected:
    friend Singleton;
    AppInfo() : _exit(false), _height(0), _width(0), _window(nullptr), _path_to_assets("assets/"){ }

public:
    static bool Running() { return !Instance()._exit || !glfwWindowShouldClose(Instance()._window); }
    static void Window_sizes(int& w, int& h)
    {
        w = Instance()._width;
        h = Instance()._height;
    }
    static GLFWwindow* Window() { return Instance()._window; }
    static void Exit() { Instance()._exit = true; }
    static std::string PathToAssets() { return Instance()._path_to_assets; }
    void set_window_size(int w, int h)
    {
        _width  = w;
        _height = h;
    }
    void set_window(GLFWwindow* window) { _window = window; }
    bool is_running();
    protected :
    int _width, _height;
    GLFWwindow* _window;
    const std::string _path_to_assets;
    bool _exit;
};