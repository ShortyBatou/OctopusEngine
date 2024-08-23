#pragma once
#include "gl_base.h"
#include <vector>
#include <Core/Base.h>
#include "Rendering/program.h"
#include "Rendering/location.h"
#include "Core/Pattern.h"
#include <cassert>
#include <string>

class GL_Program : public UniqueBinder {
public:
    explicit GL_Program(const std::string &file_path);

    virtual void bind(const Matrix4x4 &p, const Matrix4x4 &v, const Matrix4x4 &m);

    std::string program_path() { return _file_path; }

    ~GL_Program() override {
        release_program(_program);
    }

    void uniform(const char *name, const unsigned int v) const {
        glUniform1ui(location(_program, name), v);
    }

    void uniform(const char *name, const std::vector<unsigned int> &v) const {
        assert(!v.empty());
        glUniform1uiv(location(_program, name, int(v.size())), int(v.size()), v.data());
    }

    void uniform(const char *name, const int v) const {
        glUniform1i(location(_program, name), v);
    }

    void uniform(const char *name, const std::vector<int> &v) const {
        assert(!v.empty());
        glUniform1iv(location(_program, name, static_cast<int>(v.size())), static_cast<int>(v.size()), v.data());
    }

    void uniform(const char *uniform, const float v) const {
        glUniform1f(location(_program, uniform), v);
    }

    void uniform(const char *name, const std::vector<float> &v) const {
        assert(!v.empty());
        glUniform1fv(location(_program, name, static_cast<int>(v.size())), static_cast<int>(v.size()), v.data());
    }

    void uniform(const char *name, const Vector2 &v) const {
        glUniform2fv(location(_program, name), 1, &v.x);
    }

    void uniform(const char *name, const std::vector<Vector2> &v) const {
        assert(!v.empty());
        glUniform2fv(location(_program, name, static_cast<int>(v.size())), static_cast<int>(v.size()), &v[0].x);
    }

    void uniform(const char *name, const Vector3 &v) const {
        glUniform3fv(location(_program, name), 1, &v.x);
    }

    void uniform(const char *name, const std::vector<Vector3> &v) const {
        assert(!v.empty());
        glUniform3fv(location(_program, name, static_cast<int>(v.size())), static_cast<int>(v.size()), &v[0].x);
    }

    void uniform(const char *name, const Vector4 &v) const {
        glUniform4fv(location(_program, name), 1, &v.x);
    }

    void uniform(const char *name, const std::vector<Vector4> &v) const {
        assert(!v.empty());
        glUniform4fv(location(_program, name, static_cast<int>(v.size())), static_cast<int>(v.size()), &v[0].x);
    }

    void uniform(const char *name, const Matrix4x4 &v) const {
        glUniformMatrix4fv(location(_program, name), 1, GL_FALSE, &v[0][0]);
    }

    void uniform(const char *name, const std::vector<Matrix4x4> &v) const {
        glUniformMatrix4fv(location(_program, name, static_cast<int>(v.size())), static_cast<int>(v.size()), GL_FALSE,
                           &v[0][0][0]);
    }

    void program_use_texture(const char *name, int unit, GLuint texture, GLuint sampler) const;

protected:
    void bind_action() override {
        glUseProgram(_program);
    }

    void unbind_action() override {
        glUseProgram(0);
    }

    GLuint _program;
    std::string _file_path;
};
