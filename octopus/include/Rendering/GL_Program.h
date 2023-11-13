#pragma once
#include <string>
#include "gl_base.h"
#include "Rendering/program.h"
#include "Rendering/location.h"
#include "Core/Pattern.h"
#include <cassert>
class GL_Program : public UniqueBinder
{
public:
    GL_Program(const char* file_path) {
        _file_path = file_path;
        _program = read_program(file_path);
        program_print_errors(_program);
        assert(program_ready(_program));
    }

    virtual void bind(const Matrix4x4& p, const Matrix4x4& v, const Matrix4x4& m)
    { 
        this->UniqueBinder::bind();
        uniform("mvp", p * v * m);
        uniform("model", m);
        uniform("view", v);
        uniform("projection", p);
    }
    
    std::string program_path() { return _file_path; } 

    ~GL_Program() { 
        release_program(_program);
    }

    void uniform(const char* name, const unsigned int v)
    {
        glUniform1ui(location(_program, name), v);
    }

    void uniform(const char* name, const std::vector<unsigned int>& v)
    {
        assert(v.size());
        glUniform1uiv(location(_program, name, v.size()), v.size(), v.data());
    }

    void uniform(const char* name, const int v)
    {
        glUniform1i(location(_program, name), v);
    }

    void uniform(const char* name, const std::vector<int>& v)
    {
        assert(v.size());
        glUniform1iv(location(_program, name, v.size()), v.size(), v.data());
    }

    void uniform(const char* uniform, const float v)
    {
        glUniform1f(location(_program, uniform), v);
    }

    void uniform(const char* name, const std::vector<float>& v)
    {
        assert(v.size());
        glUniform1fv(location(_program, name, v.size()), v.size(), v.data());
    }

    void uniform(const char* name, const Vector2& v)
    {
        glUniform2fv(location(_program, name), 1, &v.x);
    }

    void uniform(const char* name, const std::vector<Vector2>& v)
    {
        assert(v.size());
        glUniform2fv(location(_program, name, v.size()), v.size(), &v[0].x);
    }

    void uniform(const char* name, const Vector3& v)
    {
        glUniform3fv(location(_program, name), 1, &v.x);
    }

    void uniform(const char* name, const std::vector<Vector3>& v)
    {
        assert(v.size());
        glUniform3fv(location(_program, name, v.size()), v.size(), &v[0].x);
    }

    void uniform(const char* name, const Vector4& v)
    {
        glUniform4fv(location(_program, name), 1, &v.x);
    }

    void uniform(const char* name, const std::vector<Vector4>& v)
    {
        assert(v.size());
        glUniform4fv(location(_program, name, v.size()), v.size(), &v[0].x);
    }

    void uniform(const char* name, const Matrix4x4& v)
    {
        glUniformMatrix4fv(location(_program, name), 1, GL_FALSE, &v[0][0]);
    }

    void uniform(const char* name, const std::vector<Matrix4x4>& v)
    {
        glUniformMatrix4fv(location(_program, name, v.size()), v.size(), GL_FALSE, &v[0][0][0]);
    }

    void program_use_texture(const char* name,
                             const int unit, const GLuint texture,
                             const GLuint sampler)
    {
        // verifie que l'uniform existe
        int id = location(_program, name);
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


protected:
    virtual void bind_action() override { glUseProgram(_program); }
    virtual void unbind_action() override { glUseProgram(0); }
    GLuint _program;
    const char* _file_path;
};