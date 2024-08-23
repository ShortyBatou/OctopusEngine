#include "Rendering/GL_DisplayMode.h"
#include "UI/AppInfo.h"

void GL_DisplayMode::init() {
    _graphic = this->_entity->get_component<GL_Graphic>();
    set_shaders_path(_paths);
    for (auto &_path: _paths) {
        _path = AppInfo::PathToAssets() + _path;
        _programs.push_back(new GL_Program(_path.c_str()));
    }
}

void GL_DisplayMesh::draw() {
    GL_Geometry *gl_geometry = _graphic->gl_geometry();
    for (auto &it: _graphic->gl_topologies()) {
        GL_Topology *gl_topo = it.second;

        GL_VAO *vao = gl_geometry->vao;
        GL_Buffer<Vector3> *b_vertices = gl_geometry->b_vertex;
        GL_Buffer<int> *b_line = gl_topo->b_line;
        GL_Buffer<int> *b_triangle = gl_topo->b_triangle;
        GL_Buffer<int> *b_quad = gl_topo->b_quad;

        if (b_vertices->nb_element() <= 0) return;

        _v = Camera::Instance().view();
        _p = Camera::Instance().projection();
        _pos = Camera::Instance().position();

        vao->bind();
        if (_point) draw_vertices(b_vertices);

        if (b_line->nb_element() > 0 && _wireframe) {
            draw_line(b_line);
        }

        if (b_triangle->nb_element() > 0) {
            if (_wireframe) draw_triangles_wireframe(b_triangle);
            if (_surface) {
                if (_graphic->use_element_color()) {
                    GL_Buffer<int> *b_tri_to_elem = gl_topo->sbo_tri_to_elem;
                    GL_Buffer<Color> *b_ecolors = gl_topo->sbo_ecolor;
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_tri_to_elem->gl_id());
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_ecolors->gl_id());
                }
                draw_triangles(b_triangle);
            }
        }

        if (b_quad->nb_element() > 0) {
            if (_surface) {
                if (_graphic->use_element_color()) {
                    GL_Buffer<int> *b_quad_to_elem = gl_topo->sbo_quad_to_elem;
                    GL_Buffer<Color> *b_ecolors = gl_topo->sbo_ecolor;
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, b_quad_to_elem->gl_id());
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b_ecolors->gl_id());
                }
                draw_triangles(b_quad);
            }
        }
        vao->unbind();
    }
}


void GL_DisplayMesh::set_shaders_path(std::vector<std::string> &paths) {
    // emit (no shading)
    paths.emplace_back("shaders/emit.glsl"); // 0
    paths.emplace_back("shaders/emit_vcolors.glsl"); // 1

    // flat shading
    paths.emplace_back("shaders/flat.glsl"); // 2
    paths.emplace_back("shaders/flat_vcolors.glsl"); // 3
    paths.emplace_back("shaders/flat_ecolors.glsl"); // 3
}

void GL_DisplayMesh::draw_vertices(GL_Buffer<Vector3> *b_vertices) const {
    // emit unifdraw_lineorm color or use color array buffer is multi color
    int shader_id = _graphic->use_multi_color() && !_graphic->use_element_color() && !_surface && !_wireframe;
    this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());

    if (shader_id == 0)
        this->_programs[shader_id]->uniform("color", GL_Graphic::vertice_color);

    glPointSize(GL_Graphic::vertice_size);
    glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
    glDrawArrays(GL_POINTS, 0, b_vertices->nb_element());
    glPointSize(1.f);
    this->_programs[shader_id]->unbind();
}

void GL_DisplayMesh::draw_line(GL_Buffer<int> *b_line) const {
    // emit uniform color or use color array buffer is multi color
    int shader_id = _graphic->use_multi_color() && !_graphic->use_element_color();
    _programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
    if (shader_id == 0) {
        _programs[shader_id]->uniform("color", _graphic->color() * GL_Graphic::wireframe_intencity);
    } else if (_surface && shader_id == 1) {
        _programs[shader_id]->uniform("wireframe_intencity", GL_Graphic::wireframe_intencity);
    } else if (!_surface) {
        _programs[shader_id]->uniform("wireframe_intencity", 1.f);
    }
    b_line->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(GL_Graphic::line_size);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawElements(GL_LINES, b_line->nb_element(), GL_UNSIGNED_INT, nullptr);
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.f);
    b_line->unbind();
    _programs[shader_id]->unbind();
}

void GL_DisplayMesh::draw_triangles(GL_Buffer<int> *b_triangle) const {
    int shader_id = 2;
    // use multi color or not
    shader_id += _graphic->use_multi_color();
    shader_id += _graphic->use_element_color();

    _programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
    if (shader_id == 2) {
        _programs[shader_id]->uniform("color", _graphic->color());
    }

    b_triangle->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
    glEnable(GL_LINE_SMOOTH);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, b_triangle->nb_element(), GL_UNSIGNED_INT, nullptr);
    glDisable(GL_LINE_SMOOTH);
    b_triangle->unbind();
    _programs[shader_id]->unbind();
}

void GL_DisplayMesh::draw_triangles_wireframe(GL_Buffer<int> *b_triangle) const {
    int shader_id = 0;
    shader_id += _graphic->use_multi_color() && !_graphic->use_element_color();

    this->_programs[shader_id]->bind(_p, _v, Matrix::Identity4x4());
    if (shader_id == 0)
        this->_programs[shader_id]->uniform("color", _graphic->color() * GL_Graphic::wireframe_intencity);
    else if (_surface && this->_graphic->use_multi_color()) {
        this->_programs[shader_id]->uniform("wireframe_intencity", GL_Graphic::wireframe_intencity);
    } else if (!_surface && this->_graphic->use_multi_color()) {
        this->_programs[shader_id]->uniform("wireframe_intencity", 1.f);
    }
    b_triangle->bind_to_target(GL_ELEMENT_ARRAY_BUFFER);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(2.f);
    glEnable(GL_LINE_SMOOTH);
    glDrawElements(GL_TRIANGLES, b_triangle->nb_element(), GL_UNSIGNED_INT, nullptr);
    glDisable(GL_LINE_SMOOTH);
    glLineWidth(1.f);
    b_triangle->unbind();
    this->_programs[shader_id]->unbind();
}
