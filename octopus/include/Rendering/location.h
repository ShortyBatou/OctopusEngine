#pragma once

#include "gl_base.h"
#include <stdio.h>
#include <set>
#include <string>
static int location(const GLuint program, const char* uniform,
                    const int array_size = 0)
{
    if (program == 0) return -1;

    // recuperer l'identifiant de l'uniform dans le program
    char error[4096] = {0};
    GLint location   = glGetUniformLocation(program, uniform);
    if (location < 0)
    {
#ifdef GL_VERSION_4_3
        {
            char label[1024];
            glGetObjectLabel(GL_PROGRAM, program, sizeof(label), nullptr,
                             label);

            sprintf(error, "uniform( %s %u, '%s' ): not found.", label, program,
                    uniform);
        }
#else
        sprintf(error, "uniform( program %u, '%s'): not found.", program,
                uniform);
#endif

        static std::set<std::string> log;
        if (log.insert(error).second == true)
            // pas la peine d'afficher le message 60 fois par seconde...
            printf("%s\n", error);

        return -1;
    }

#ifndef GK_RELEASE
    // verifier que le program est bien en cours d'utilisation, ou utiliser
    // glProgramUniform, mais c'est gl 4
    GLuint current;
    glGetIntegerv(GL_CURRENT_PROGRAM, (GLint*)&current);
    if (current != program)
    {
#ifdef GL_VERSION_4_3
        {
            char label[1024];
            glGetObjectLabel(GL_PROGRAM, program, sizeof(label), nullptr,
                             label);
            char labelc[1024];
            glGetObjectLabel(GL_PROGRAM, current, sizeof(labelc), nullptr,
                             labelc);

            sprintf(error,
                    "uniform( %s %u, '%s' ): invalid current shader program( "
                    "%s %u )",
                    label, program, uniform, labelc, current);
        }
#else
        sprintf(error,
                "uniform( program %u, '%s' ): invalid current shader program( "
                "%u )...",
                program, uniform, current);
#endif

        printf("%s\n", error);
        glUseProgram(program);
    }

    if (location >= 0 && array_size > 0)
    {
#ifdef GL_VERSION_4_3
        // verifier que le tableau d'uniform fait la bonne taille...
        GLuint index = glGetProgramResourceIndex(program, GL_UNIFORM, uniform);
        if (index != GL_INVALID_INDEX)
        {
            GLenum props[] = {GL_ARRAY_SIZE};
            GLint value    = 0;
            glGetProgramResourceiv(program, GL_UNIFORM, index, 1, props, 1,
                                   nullptr, &value);
            if (value != array_size)
            {
                char label[1024];
                glGetObjectLabel(GL_PROGRAM, program, sizeof(label), nullptr,
                                 label);

                printf("uniform( %s %u, '%s' array [%d] ): invalid array size "
                       "[%d]...\n",
                       label, location, uniform, value, array_size);
            }
        }
#endif
    }
#endif

    return location;
}
