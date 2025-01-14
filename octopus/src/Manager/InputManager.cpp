#pragma once
#include "Core/Base.h"
#include "UI/AppInfo.h"
#include "imgui.h"
#include "Manager/Input.h"

#include "Manager/InputManager.h"

void key_callback(GLFWwindow *, const int key, int, const int action, int) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (action == GLFW_PRESS) Input::Instance().keyboard().set_pressed(static_cast<Key>(key));
    if (action == GLFW_RELEASE) Input::Instance().keyboard().set_released(static_cast<Key>(key));
}

void mouse_callback(GLFWwindow *, const double xpos, const double ypos) {
    ImGui::GetIO().AddMousePosEvent(static_cast<scalar>(xpos), static_cast<scalar>(ypos));

    if (ImGui::GetIO().WantCaptureMouse) return;
    Input::Instance().set_mouse_position(Vector2(static_cast<scalar>(xpos), static_cast<scalar>(ypos)));
}

void scroll_callback(GLFWwindow *, const double xoffset, const double yoffset) {
    ImGuiIO &io = ImGui::GetIO();
    io.AddMouseWheelEvent(static_cast<scalar>(xoffset), static_cast<scalar>(yoffset));

    if (ImGui::GetIO().WantCaptureMouse) return;
    Input::Instance().set_scroll(Vector2(static_cast<scalar>(xoffset), static_cast<scalar>(yoffset)));
}

void mouse_button_callback(GLFWwindow *, int button, int action, int) {
    ImGuiIO &io = ImGui::GetIO();
    io.AddMouseButtonEvent(button, action);
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (action == GLFW_PRESS) Input::Instance().mouse().set_pressed(static_cast<Mouse>(button));
    if (action == GLFW_RELEASE) Input::Instance().mouse().set_released(static_cast<Mouse>(button));
}

InputManager::InputManager() {
    Input::Instance();
    glfwSetKeyCallback(AppInfo::Window(), key_callback);
    glfwSetCursorPosCallback(AppInfo::Window(), mouse_callback);
    glfwSetScrollCallback(AppInfo::Window(), scroll_callback);
    glfwSetMouseButtonCallback(AppInfo::Window(), mouse_button_callback);
}

void InputManager::update() {
    Input::Instance().update_devices();
}

InputManager::~InputManager() {
    Input::Delete();
}
