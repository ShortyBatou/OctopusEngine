#pragma once
#include "Core/Pattern.h"
#include <GL/glew.h>
#include <vector>

template<typename T>
struct GL_Buffer final : UniqueBinder {
    explicit GL_Buffer(GLenum default_target, GLenum type = GL_STREAM_DRAW);

    void resize(int nb_element);

    void load_data(const std::vector<T> &data);

    void bind_to_target(GLuint target) {
        _current_target = target;
        bind();
    }

    [[nodiscard]] GLuint gl_id() const { return _gl_id; }
    bool binded() override { return _binded; }
    [[nodiscard]] GLenum default_target() const { return _default_target; }
    [[nodiscard]] GLenum current_target() const { return _current_target; }
    [[nodiscard]] GLenum type() const { return _type; }
    [[nodiscard]] GLsizei nb_element() const { return _nb_element; }

    void set_default_target(GLuint target) {
        assert(!this->_binded);
        _default_target = target;
    }

    ~GL_Buffer() override { release(); }

protected:
    void bind_action() override {
        glBindBuffer(_current_target, _gl_id);
    }

    void unbind_action() override {
        glBindBuffer(_current_target, 0);
        _current_target = _default_target;
    }

    void generate() {
        glGenBuffers(1, &_gl_id);
    }

    void release() {
        assert(!this->_binded);
        glDeleteBuffers(1, &_gl_id);
    }

    GLsizei _nb_element;
    GLenum _default_target;
    GLenum _current_target;
    GLenum _type;
    GLuint _gl_id{};
};

struct GL_VAO final : UniqueBinder {
    GL_VAO() : _id(0) {
        glGenVertexArrays(1, &_id);
    }

    template<typename T>
    void bind_array(GL_Buffer<T> *buffer, int bind_index, int nb_data, GLuint data_type) {
        buffer->bind();
        glVertexAttribPointer(bind_index, nb_data, data_type, GL_FALSE, 0, 0);
        buffer->unbind();
    }

    void enable_attribute(GLuint attribute_id) const {
        assert(_binded);
        glEnableVertexAttribArray(attribute_id);
    }

    void disable_attribute(GLuint attribute_id) const {
        assert(this->_binded);
        glDisableVertexAttribArray(attribute_id);
    }

    ~GL_VAO() override { release(); }

    [[nodiscard]] GLuint id() const { return _id; }

protected:
    void release() const {
        assert(!this->_binded);
        glDeleteVertexArrays(1, &_id);
    }

    void bind_action() override {
        glBindVertexArray(_id);
    }

    void unbind_action() override {
        glBindVertexArray(0);
    }

    GLuint _id;
};
