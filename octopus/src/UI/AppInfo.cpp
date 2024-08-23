#include "UI/AppInfo.h"
#include <GLFW/glfw3.h>

bool AppInfo::Running()
{
    return !Instance()._exit || !glfwWindowShouldClose(Instance()._window);
}
