#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Manager/Input.h"
#include "UI/AppInfo.h"
#include <GLFW/glfw3.h>

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (action == GLFW_PRESS)   Input::Instance().keyboard().set_pressed((Key)key);
    if (action == GLFW_RELEASE) Input::Instance().keyboard().set_released((Key)key);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(scalar(xpos), scalar(ypos));

    if (ImGui::GetIO().WantCaptureMouse) return;
    Input::Instance().set_mouse_position(Vector2(scalar(xpos), scalar(ypos)));
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) { 
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseWheelEvent(scalar(xoffset), scalar(yoffset));

    if (ImGui::GetIO().WantCaptureMouse) return;
    Input::Instance().set_scroll(Vector2(scalar(xoffset), scalar(yoffset)));
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseButtonEvent(button, action);

    if (ImGui::GetIO().WantCaptureMouse) return;
    if (action == GLFW_PRESS)   Input::Instance().mouse().set_pressed((Mouse)button);
    if (action == GLFW_RELEASE) Input::Instance().mouse().set_released((Mouse)button);
}

class InputManager : public Behaviour
{
public:
    InputManager() { 
        Input::Instance();
        glfwSetKeyCallback(AppInfo::Window(), key_callback);
        glfwSetCursorPosCallback(AppInfo::Window(), mouse_callback);
        glfwSetScrollCallback(AppInfo::Window(), scroll_callback);
        glfwSetMouseButtonCallback(AppInfo::Window(), mouse_button_callback);
    }
    virtual void update() override { 
        Input::Instance().update_devices();
    }

    virtual ~InputManager() { 
        Input::Delete();
    }
};

