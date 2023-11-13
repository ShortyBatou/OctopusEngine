/**
 * https://learnopengl.com/Introduction
 */

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "HUD/Application.h"

int main()
{
    Application* app = new Application();
    app->init();
    // MAIN LOOP
    while (AppInfo::Running())
    {   
        /// ENGINE
        app->update();
    }

    delete app;
    return EXIT_SUCCESS;
}