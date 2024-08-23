#include "Rendering\GL_Buffer.h"

template<typename T>
GL_Buffer<T>::GL_Buffer(GLenum default_target, GLenum type)
    : _nb_element(0)
      , _type(type)
      , _default_target(default_target)
      , _current_target(default_target) { generate(); }

template<typename T>
void GL_Buffer<T>::resize(int nb_element) {
    _nb_element = nb_element;
    bind_to_target(_default_target);
    glBufferData(_default_target, _nb_element * sizeof(T), nullptr, _type);
    unbind();
}

template<typename T>
void GL_Buffer<T>::load_data(const std::vector<T> &data) {
    _nb_element = data.size();
    bind_to_target(_default_target);
    glBufferData(_default_target, _nb_element * sizeof(T), data.data(), _type);
    unbind();
}
