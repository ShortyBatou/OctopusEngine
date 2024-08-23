#pragma once
#include "Rendering/gl_base.h"

#include <string>
#include <fstream>

// charge un fichier texte.
std::string read(const char* filename);

// insere les definitions apres la ligne contenant #version
std::string prepare_source(std::string file, const std::string& definitions);
const char* shader_string(GLenum type);

static const char* shader_keys[] = {"VERTEX_SHADER", "FRAGMENT_SHADER", "GEOMETRY_SHADER", "TESSELATION_CONTROL", "EVALUATION_CONTROL", "COMPUTE_SHADER"};
static const int shader_keys_max = 6;

static GLenum shader_types[]
    = {GL_VERTEX_SHADER,       GL_FRAGMENT_SHADER,        GL_GEOMETRY_SHADER,
#ifdef GL_VERSION_4_0
       GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER,
#else
    0, 
    0,
#endif
#ifdef GL_VERSION_4_3
       GL_COMPUTE_SHADER
#else
    0
#endif
};

GLuint compile_shader(GLuint program, GLenum shader_type, const std::string& source);

int reload_program(GLuint program, const char* filename,const char* definitions);

GLuint read_program(const char* filename, const char* definitions = "");

int release_program(GLuint program);

bool program_ready(GLuint program);

bool program_errors(GLuint program);

// formatage des erreurs de compilation des shaders

void print_line(std::string& errors, const char* source, int begin_id, int line_id);
static int print_errors(std::string& errors, const char* log, const char* source);

int program_format_errors(GLuint program, std::string& errors);

int program_print_errors(GLuint program);
