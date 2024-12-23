#include "Script/VTK//VTK_FEM.h"
#include "Manager/Input.h"
#include "Rendering/GL_Graphic.h"

void VTK_Graphic::late_update() {
    if (Input::Down(Key::S) && Input::Loop(Key::LEFT_CONTROL)) {
        save();
    }
}

void VTK_Graphic::save() const {
    GL_Graphic* graphic = this->_entity->get_component<GL_Graphic>();
    VTK_Formater vtk;

    //std::map<Element, Mesh::Topology> lines;
    //lines[Line] = graphic->get_lines();
    //vtk.open(_name + "_Line");
    //vtk.save_mesh(graphic->get_geometry(), lines);
    //vtk.close();

    //if (this->_entity->get_component<GL_GraphicHighOrder>()) {
    //    std::map<Element, Mesh::Topology> mesh;
    //    mesh[Triangle] = graphic->get_quads();
    //    vtk.open(_name);
    //    vtk.save_mesh(graphic->get_geometry(), mesh);
    //    vtk.close();
    //}
    //else {
    //    std::map<Element, Mesh::Topology> mesh;
    //    mesh[Triangle] = graphic->get_triangles();
    //    mesh[Quad] = graphic->get_quads();
    //    vtk.open(_name);
    //    vtk.save_mesh(graphic->get_geometry(), mesh);
    //    vtk.close();
    //}
}
