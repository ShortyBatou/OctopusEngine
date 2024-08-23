/**
 * https://learnopengl.com/Introduction
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <UI/AppInfo.h>

#include "UI/Application.h"

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