#pragma once
#include "Core/Base.h"
#include "Manager/Input.h"
#include "Mesh/Converter/VTK_Formater.h"
#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"
#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_GraphicHighOrder.h"

class VTK_Graphic : public Component {
public:
    VTK_Graphic(std::string name) : _name(name) { }

    virtual void late_update() {
        if (Input::Down(Key::S) && Input::Loop(Key::LEFT_CONTROL)) {
            save();
        }
    }

    void save() {
        GL_Graphic* graphic = this->_entity->getComponent<GL_Graphic>();
        VTK_Formater vtk;

        std::map<Element, Mesh::Topology> lines;
        lines[Line] = graphic->get_lines();
        vtk.open(_name + "_Line");
        vtk.save_mesh(graphic->get_geometry(), lines);
        vtk.close();

        if (this->_entity->getComponent<GL_GraphicHighOrder>()) {
            std::map<Element, Mesh::Topology> mesh;
            mesh[Triangle] = graphic->get_quads();
            vtk.open(_name);
            vtk.save_mesh(graphic->get_geometry(), mesh);
            vtk.close();
        }
        else {
            std::map<Element, Mesh::Topology> mesh;
            mesh[Triangle] = graphic->get_triangles();
            mesh[Quad] = graphic->get_quads();
            vtk.open(_name);
            vtk.save_mesh(graphic->get_geometry(), mesh);
            vtk.close();
        }

    }

    std::string file_name() {
        return _name;
    }
protected:
    std::string _name;
};