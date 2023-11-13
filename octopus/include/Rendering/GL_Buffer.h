#pragma once
#include "Core/Pattern.h"
#include <GL/glew.h>
#include <vector>

template<typename T>
struct GL_Buffer : public UniqueBinder
{
    GL_Buffer(GLenum default_target, GLenum type = GL_STREAM_DRAW ) 
        : _nb_element(0)
        , _type(type)
        , _default_target(default_target)
        , _current_target(default_target)
    { generate(); }

    void resize(unsigned int nb_element)
    {
        _nb_element = nb_element;
        bind_to_target(_default_target);
        glBufferData(_default_target, _nb_element * sizeof(T), nullptr, _type);
        unbind();
    }

    void load_data(const std::vector<T>& data) {
        _nb_element = data.size();
        bind_to_target(_default_target);
        glBufferData(_default_target, _nb_element * sizeof(T), data.data(), _type);
        unbind();
    }

    void bind_to_target(GLuint target)
    {
        _current_target = target;
        bind();
    }

    GLuint gl_id() { return _gl_id; }
    bool binded() { return _binded; }
    GLenum default_target() { return _default_target; }
    GLenum current_target() { return _current_target; }
    GLenum type() { return _type; }
    GLsizei nb_element() { return _nb_element; }

    void set_default_target(GLuint target)
    {
        assert(!this->_binded);
        _default_target = target;
    }
    virtual ~GL_Buffer() { release(); }

protected:
    virtual void bind_action() override
    {
        glBindBuffer(_current_target, _gl_id);
    }
    virtual void unbind_action() override
    {
        glBindBuffer(_current_target, 0);
        _current_target = _default_target;
    }
    void generate() { glGenBuffers(1, &_gl_id); }
    void release()
    {
        assert(!this->_binded);
        glDeleteBuffers(1, &_gl_id);
    }
    
    GLsizei _nb_element;
    GLenum _default_target;
    GLenum _current_target;
    GLenum _type;
    GLuint _gl_id;
};

struct GL_VAO : public UniqueBinder
{
    GL_VAO() : _id(0) { glGenVertexArrays(1, &_id); }

    template<typename T>
    void bind_array(GL_Buffer<T>* buffer, unsigned int bind_index, unsigned int nb_data, GLuint data_type)
    {
        buffer->bind();
        glVertexAttribPointer(bind_index, nb_data, data_type, GL_FALSE, 0, 0);
        buffer->unbind();
    }

    void enable_attribute(GLuint attribute_id) {
        assert(_binded);
        glEnableVertexAttribArray(attribute_id);
    }

    void disable_attribute(GLuint attribute_id) {
        assert(this->_binded);
        glDisableVertexAttribArray(attribute_id);
    }

    virtual ~GL_VAO() { release(); }

    GLuint id() { return _id; }
protected:
    void release()
    {
        assert(!this->_binded);
        glDeleteVertexArrays(1, &_id);
    }
    virtual void bind_action()   override { glBindVertexArray(_id); }
    virtual void unbind_action() override
    {
        glBindVertexArray(0);
    }
    GLuint _id;
};