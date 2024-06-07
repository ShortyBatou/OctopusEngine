#pragma once
#include "Scene.h"
#include "Manager/TimeManager.h"
#include "Manager/OpenglManager.h"
#include "Manager/CameraManager.h"
#include "Manager/InputManager.h"
#include "Manager/DebugManager.h"

#include "Mesh/Mesh.h"
#include "Mesh/Generator/PrimitiveGenerator.h"
#include "Mesh/Generator/BeamGenerator.h"
#include "Mesh/Converter/MeshLoader.h"

#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_GraphicElement.h"
#include "Rendering/GL_DisplayMode.h"
#include "Rendering/GL_GraphicSurface.h"
#include "Rendering/GL_GraphicHighOrder.h"

#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/FEM_Dynamic.h"
#include "Script/Dynamic/Constraint_Rigid_Controller.h"
#include "Script/Dynamic/ConstantForce_Controller.h"
#include "Script/VTK/VTK_FEM.h"
#include "Script/Record/DataRecorder.h"
#include "UI/UI_Component.h"

struct MeshScene : public Scene
{
    virtual char* name() override { return "Mesh Scene"; }

    virtual void init() override {

    }

    virtual void build_editor(UI_Editor* editor) {
        editor->add_manager_ui(new UI_Time());
        editor->add_manager_ui(new UI_DisplaySettings());
        editor->add_manager_ui(new UI_Camera());
        editor->add_component_ui(new UI_Mesh_Display());
        editor->add_component_ui(new UI_Data_Recorder());
        editor->add_component_ui(new UI_Graphic_Saver());
        editor->add_component_ui(new UI_PBD_Dynamic());
        editor->add_component_ui(new UI_Constraint_Rigid_Controller());
    }

    virtual void build_root(Entity* root) override
    {
        root->add_behaviour(new TimeManager(1.f / 60.f));
        root->add_behaviour(new InputManager());
        root->add_behaviour(new CameraManager());
        root->add_behaviour(new DebugManager(false));
        root->add_behaviour(new OpenGLManager(Color(0.9f, 0.9f, 0.9f, 1.f)));
    }

    // build scene's entities
    virtual void build_entities() override
    {
        Vector3 size(1, 1, 1);
        Vector3I cells;

        cells = Vector3I(1, 1, 1);
        build_beam_mesh(Vector3(0, 0, 0), cells, size, Color(0.4f, 0.4f, 0.8f, 1.f), Tetra20);
        //build_beam_mesh(Vector3(0, 0, 0), cells, size, Color(0.8, 0.3, 0.8, 1.), Hexa);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.3, 0.3, 0.8, 1.), Tetra, false, false);
        //cells = Vector3I(8, 3, 3);
        //build_vtk_mesh(Vector3(0, 0, 0), cells, size, Color(0.3, 0.3, 0.8, 1.), "mesh/vtk/hexa.vtk");
        //convert_vtk_mesh("Tetra10_8_2_2_4x1x1.vtk", "Torsion_", Tetra10, 4);
        //cells = Vector3I(6, 2, 2);
        //build_xpbd_entity(Vector3(0, 0, 2), cells, size, Color(0.8, 0.3, 0.3, 1.), Tetra, false, true);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra10, true, false);
    }

    Mesh* get_sphere_mesh(const Vector3& pos, Element& element) {
        MeshGenerator* generator = new TetraSphere(element);
        generator->setTransform(glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = generator->build();
        delete generator;
        return mesh;
    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, Element element) {
        BeamMeshGenerator* generator;
        switch (element)
        {
        case Tetra: generator = new TetraBeamGenerator(cells, size); break;
        case Pyramid: generator = new PyramidBeamGenerator(cells, size); break;
        case Prism: generator = new PrismBeamGenerator(cells, size); break;
        case Hexa: generator = new HexaBeamGenerator(cells, size); break;
        case Tetra10: generator = new TetraBeamGenerator(cells, size); break;
        case Tetra20: generator = new TetraBeamGenerator(cells, size); break;
        default: break;
        }
        generator->setTransform(glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = generator->build();

        delete generator;
        return mesh;
    }

    void build_msh_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, std::string file) {
        Msh_Loader loader(AppInfo::PathToAssets() + file);
        loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = loader.build();
        build_entity(mesh, color);
    }

    void convert_vtk_mesh(std::string file, std::string name, Element s_elem, int sub) {
        VTK_Loader loader(AppInfo::PathToAssets() + file);
        Mesh* mesh = loader.build();
        std::vector<Vector3> u_v3 = loader.get_point_data_v3("u");
        MeshMap* map = tetra_to_linear(mesh, s_elem, sub);
        std::vector<Vector3> u_v3_refined = map->convert<Vector3>(mesh, u_v3);
        map->apply_to_mesh(mesh);

        VTK_Formater vtk;
        vtk.open(name + std::string(element_name(s_elem)) + "_to_T4_R" + std::to_string(sub));
        vtk.save_mesh(mesh->geometry(), mesh->topologies());
        vtk.start_point_data();
        vtk.add_vector_data(u_v3_refined, "u");
        vtk.close();
    }

    void build_vtk_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, std::string file) {
        VTK_Loader loader(AppInfo::PathToAssets() + file);
        loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = loader.build();
        std::vector<Vector3> u_v3 = loader.get_point_data_v3("u");
        for (int i = 0; i < mesh->geometry().size(); ++i) {
            mesh->geometry()[i] += u_v3[i];
        }
        build_entity(mesh, color);
    }

    void build_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element) {
        Mesh* mesh;
        //mesh = get_beam_mesh(pos, cells, size, element);
        mesh = get_sphere_mesh(pos, element);
        //if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        //if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
        mesh->set_dynamic_geometry(true);
        build_entity(mesh, color);
    }

    void build_entity(Mesh* mesh, const Color& color) {
        Entity* e = Engine::CreateEnity();
        e->add_behaviour(mesh);

        // Mesh converter simulation to rendering (how it will be displayed)
        GL_Graphic* graphic;
        int type = (mesh->topologies()[Tetra10].size() != 0 || mesh->topologies()[Tetra20].size() != 0) ? 2 : 0;
        //graphic = new GL_GraphicElement(0.7);
        switch (type) {
            case 0: graphic = new GL_GraphicSurface(color); break;
            //case 1 : graphic = new GL_GraphicElement(0.7f); break;
            //case 2 : graphic = new GL_GraphicHighOrder(5, color); break;
            default: graphic = new GL_GraphicSurface(color); break;
        }

        e->add_component(graphic);

        // Opengl Rendering
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->wireframe() = true;
        display->point() = false;
        e->add_component(display);

        // save mesh in VTK format (Paraview)

        std::string file_name = "Mesh";
        DataRecorder* data_recorder = new DataRecorder(file_name);
        data_recorder->add(new MeshRecorder());
        data_recorder->add(new Mesh_VTK_Recorder(file_name));
        data_recorder->add(new Graphic_VTK_Recorder(file_name));
        e->add_component(data_recorder);
    }
};