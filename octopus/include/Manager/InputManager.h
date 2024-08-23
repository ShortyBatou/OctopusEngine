#pragma once
#include <GLFW/glfw3.h>

#include "Core/Pattern.h"

void key_callback(GLFWwindow *, int key, int, int action, int);

void mouse_callback(GLFWwindow *, double xpos, double ypos);

void scroll_callback(GLFWwindow *, double xoffset, double yoffset);

void mouse_button_callback(GLFWwindow *, int button, int action, int);

class InputManager final : public Behaviour {
public:
    InputManager();

    void update() override;

    ~InputManager() override;
};
