#include "Rendering/GL_Program.h"
#include <cassert>

GL_Program::GL_Program(const std::string &file_path) {
    _file_path = file_path;
    _program = read_program(file_path.c_str());
    program_print_errors(_program);
    assert(program_ready(_program));
}

void GL_Program::bind(const Matrix4x4 &p, const Matrix4x4 &v, const Matrix4x4 &m) {
    this->UniqueBinder::bind();
    uniform("mvp", p * v * m);
    uniform("model", m);
    uniform("view", v);
    uniform("projection", p);
}

void GL_Program::program_use_texture(const char *name, const int unit, const GLuint texture,
                                     const GLuint sampler) const {
    // verifie que l'uniform existe
    const int id = location(_program, name);
    if (id < 0) return;

    // selectionne l'unite de texture
    glActiveTexture(GL_TEXTURE0 + unit);
    // configure la texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // les parametres de filtrage
    glBindSampler(unit, sampler);

    // transmet l'indice de l'unite de texture au shader
    glUniform1i(id, unit);
}
