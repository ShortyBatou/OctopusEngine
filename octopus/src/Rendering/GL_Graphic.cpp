#include "Rendering/GL_Graphic.h"

/// Vertex data for display

GL_Geometry::GL_Geometry() {
    vao = new GL_VAO();
    b_vertex = new GL_Buffer<Vector3>(GL_ARRAY_BUFFER);
    b_color = new GL_Buffer<Vector4>(GL_ARRAY_BUFFER);
    b_normal = new GL_Buffer<Vector3>(GL_ARRAY_BUFFER);
    init_vao();
}

void GL_Geometry::init_vao() const {
    vao->bind();
    vao->bind_array(b_vertex, 0, 3, GL_FLOAT);
    vao->bind_array(b_color, 1, 4, GL_FLOAT);
    vao->bind_array(b_normal, 2, 3, GL_FLOAT);
    vao->enable_attribute(0);
    vao->enable_attribute(1);
    vao->enable_attribute(2);
    vao->unbind();
}

void GL_Geometry::clear() {
    geometry.clear();
}

void GL_Geometry::load() const {
    b_vertex->load_data(geometry);
}

void GL_Geometry::load_colors() const
{
    assert(vcolors.size() == geometry.size());
    b_color->load_data(vcolors);
}



GL_Topology::GL_Topology() {
    b_line = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    b_triangle = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    b_quad = new GL_Buffer<int>(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    sbo_tri_to_elem = new GL_Buffer<int>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
    sbo_quad_to_elem = new GL_Buffer<int>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
    sbo_ecolor = new GL_Buffer<Color>(GL_SHADER_STORAGE_BUFFER, GL_STATIC_DRAW);
}

GL_Topology::~GL_Topology() {
    delete b_line;
    delete b_triangle;
    delete b_quad;
    delete sbo_tri_to_elem;
    delete sbo_quad_to_elem;
}

void GL_Topology::load()
{
    if (!lines.empty())
        b_line->load_data(lines);
    if (!triangles.empty())
        b_triangle->load_data(triangles);
    if (!quads.empty())
        b_quad->load_data(quads);
    if (!tri_to_elem.empty())
        sbo_tri_to_elem->load_data(tri_to_elem);
    if (!quad_to_elem.empty())
        sbo_quad_to_elem->load_data(quad_to_elem);
}

void GL_Topology::laod_ecolor() {
    if (!ecolors.empty())
        sbo_ecolor->load_data(ecolors);
}

void GL_Topology::clear() {
    lines.clear();
    triangles.clear();
    quads.clear();
    tri_to_elem.clear();
    quad_to_elem.clear();
}


void GL_Graphic::init()
{
    _mesh = entity()->get_component<Mesh>();
}

void GL_Graphic::late_init() {
    update();
}

void GL_Graphic::update()
{

    if (_mesh->need_update() || _mesh->has_dynamic_topology())
    {
        // get and clear the previous displayed topology
        // if gl topo not found create a new one
        for (auto& it : _mesh->topologies()) {
            Element element = it.first;
            if (_gl_topologies.find(element) == _gl_topologies.end()) {
                _gl_topologies[element] = new GL_Topology();
            }
            GL_Topology* gl_topo = _gl_topologies[element];
            gl_topo->clear();
        }

        // update displayed topology
        update_gl_topology();

        // load it to buffers
        for (auto& it : _mesh->topologies()) {
            _gl_topologies[it.first]->load();
        }
    }

    if (_mesh->need_update() || _mesh->has_dynamic_geometry())
    {
        _gl_geometry->clear();
        update_gl_geometry();
        _gl_geometry->load();
    }

    if (_mesh->need_update() || _mesh->has_dynamic_geometry() || _mesh->has_dynamic_topology()) {
        // if the number of geometry change, we need to change the colors array buffer
        if (_multi_color && !_element_color) {
            update_gl_vcolors();
            _gl_geometry->load_colors();
        } // else element color is directly handled by topology update, no change is needded
        else if (_multi_color && _element_color) {
            for (auto& it : _mesh->topologies()) {
                _gl_topologies[it.first]->laod_ecolor();
            }
        }
    }

    _mesh->need_update() = false;
}


GL_Graphic::~GL_Graphic() {
    delete _gl_geometry;
    for (auto& it : _gl_topologies) {
        delete it.second;
    }
    _gl_topologies.clear();
}

void GL_Graphic::update_gl_topology() {
    for (auto& it : _mesh->topologies()) {
        Element element = it.first;
        if (element != Line && element != Triangle && element != Quad) continue;

        GL_Topology* gl_topo = _gl_topologies[element];
        switch (element)
        {
            case Line: gl_topo->lines = it.second; break;
            case Triangle: gl_topo->triangles = it.second; break;
            case Quad: gl_topo->quads = it.second; break;
            default: break;
        }
        if (element == Line) continue;

        Mesh::Topology& face_to_elem = (element == Triangle) ? gl_topo->tri_to_elem : gl_topo->quad_to_elem;
        const int nb = elem_nb_vertices(element);
        const int nb_element = static_cast<int>(it.second.size()) /nb;
        face_to_elem.resize(nb_element);
        std::iota(std::begin(face_to_elem), std::end(face_to_elem), 0); // (0,1,2,3, ..., n-1)
    }
}

scalar GL_Graphic::vertice_size = 2.f;
scalar GL_Graphic::line_size = 2.f;
scalar GL_Graphic::wireframe_intencity = 0.7f;
Color GL_Graphic::vertice_color = ColorBase::Grey(0.1f);