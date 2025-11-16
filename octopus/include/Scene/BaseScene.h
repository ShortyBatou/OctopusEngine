#pragma once
#include <dinput.h>
#include <Mesh/Converter/MeshLoader.h>
#include <Script/Dynamic/Cuda_Constraint_Rigid_Controller.h>
#include <Script/Dynamic/Cuda_LF_VBD_FEM_Dynamic.h>
#include <Script/Dynamic/Cuda_MG_VBD_FEM_Dynamic.h>
#include <Script/Dynamic/Cuda_VBD_FEM_Dynamic.h>
#include <Script/Dynamic/MG_VBD_FEM_Dynamic.h>
#include <Script/Dynamic/XPBD_ShapeMatching_Dynamic.h>
#include <Script/Dynamic/Cuda_Mixed_VBD_FEM_Dynamic.h>
#include "Scene.h"
#include "Manager/TimeManager.h"
#include "Manager/OpenglManager.h"
#include "Manager/CameraManager.h"
#include "Manager/InputManager.h"
#include "Manager/DebugManager.h"
#include "Manager/Dynamic.h"
#include "Tools/Area.h"
#include "Mesh/Mesh.h"
#include "Mesh/Generator/BeamGenerator.h"

#include "Rendering/GL_Graphic.h"
#include "Rendering/GL_DisplayMode.h"
#include "Rendering/GL_GraphicSurface.h"
#include "Rendering/GL_GraphicHighOrder.h"

#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "UI/UI_Component.h"

#include "Script/Dynamic/Cuda_XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/VBD_FEM_Dynamic.h"
#include "Script/Measure/MeshDiff.h"
#include "Script/VTK/VTK_Attribute.h"

struct SimulationArgs {
    scalar density;
    Mass_Distribution distribution;
    scalar young;
    scalar poisson;
    scalar damping;
    Material material;
    FEM_DataDisplay::Type display;
    int iteration;
    int sub_iteration;
    int scenario_1;
    int scenario_2;
    std::string mesh_file;
    std::string mesh_type;
    bool biased = false; // only usefull with tetra beam generated mesh
    int refine;
    Vector3 dir;
};

struct BaseScene final : Scene
{
    char* name() override { return "Basic Scene"; }

    void build_editor(UI_Editor* editor) override {
        editor->add_manager_ui(new UI_Time());
        editor->add_manager_ui(new UI_Dynamic());
        editor->add_manager_ui(new UI_DisplaySettings());
        editor->add_manager_ui(new UI_Camera());

        editor->add_component_ui(new UI_Mesh_Display());
        editor->add_component_ui(new UI_Cuda_Constraint_Rigid_Controller());
        editor->add_component_ui(new UI_Data_Recorder());
        editor->add_component_ui(new UI_Data_Displayer());
        editor->add_component_ui(new UI_Graphic_Saver());
    }

    void build_root(Entity* root) override
    {
        root->add_behaviour(new TimeManager(1.f / 60.f));
        root->add_behaviour(new DynamicManager(Vector3(0.,-9.81*0.f,0.)));
        root->add_behaviour(new InputManager());
        root->add_behaviour(new CameraManager());
        root->add_behaviour(new DebugManager(true));
        root->add_behaviour(new OpenGLManager(Color(1.0f,1.0f,1.0f,1.f)));
        //root->add_behaviour(new MeshDiff_MSE(1,{2}));
        //root->add_behaviour(new Beam_MSE_Sampling(1, {2}, 5));
    }

    // build scene's entities
    void build_entities() override
    {
        SimulationArgs args{};
        args.density = 1000; args.distribution = FemShape;
        args.young = 1e6; args.poisson = 0.45; args.material = Stable_NeoHooke;
        args.damping = 5e-6;
        args.iteration = 1; args.sub_iteration = 1  ;
        args.scenario_1 = 0; args.scenario_2 = 5; args.dir = Unit3D::right();
        args.display = FEM_DataDisplay::Type::Displacement;
        //args.mesh_file = "mesh/vtk/bunny_Q1.vtk";
        //args.mesh_type = "vtk";
        //args.mesh_file = "mesh/msh/bar_tetra_1300.msh";
        //args.mesh_type = "msh";
        Vector3I cells(12, 3, 3);

        Vector3 size(4, 1, 1);
        //args.mesh_file = "1_Hexa_16_16_64_2x2x8_400.vtk";
        //args.mesh_type = "vtk";
        //build_mesh_entity(Vector3(-1,-1.,0), cells, size, Color(0.8,.2,0.8,0), Hexa, args);
        //args.mesh_type = "";

        args.damping = 5e-6;
        args.biased = false;
        //args.sub_iteration = 500; build_fem_entity(Vector3(0,0.,-4), cells, size, Color(0.8,.2,0.8,0), Hexa, args, true);

        size = Vector3I(4, 1, 1);
        args.iteration = 1;
        args.sub_iteration = 1;
        args.damping = 40;
        args.refine = 0;
        //for(int i = 1; i <= 4; ++i) {
            //args.sub_iteration = 120 + i*20;
            //args.material = Stable_NeoHooke; build_xpbd_entity(Vector3(0,0.,0), cells, size, Color(0.8,.2,0.8,0), Tetra, args, true, false);
        args.sub_iteration = 100; build_xpbd_entity(Vector3(0,0,0), cells, size, Color(0.8,.2,0.8,0), Hexa, args, true, false);
        //args.sub_iteration = 100; build_xpbd_entity(Vector3(0,0,0), cells, size, Color(0.8,.2,0.8,0), Tetra20, args, true, false);
        //args.sub_iteration = 100; build_xpbd_entity(Vector3(0,0,0), cells, size, Color(0.8,.2,0.8,0), Hexa, args, true, false);
        //args.sub_iteration = 100; build_xpbd_entity(Vector3(0,0,0), cells, size, Color(0.8,.2,0.8,0), Hexa27, args, true, false);


        //args.sub_iteration = 50; args.material = NeoHooke; build_xpbd_entity(Vector3(0,0.,0), cells, size, Color(0.8,.2,0.8,0), Tetra, args, true, false);
        //args.sub_iteration = 20; args.material = Stable_NeoHooke; build_xpbd_entity(Vector3(0,0.,0), cells, size, Color(0.8,.2,0.8,0), Tetra, args, true, false);
        //args.sub_iteration = 50; args.material = Stable_NeoHooke; build_xpbd_entity(Vector3(0,0.,0), cells, size, Color(0.8,.2,0.8,0), Tetra, args, true, false);
            //build_fem_entity(Vector3(0,0.,0), cells, size, Color(0.8,.2,0.8,0), Tetra, args, true);
        //build_vbd_entity(Vector3(0,0,0), cells, size, Color(0.8,.2,0.8,0), Hexa27, args, 0., true);

        //}



        //build_xpbd_entity(Vector3(0,0.,0), cells, size, Color(0.8,.2,0.8,0), Tetra, args, true, false);
    }

    Mesh* get_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Element element, const bool biased) {
        BeamMeshGenerator* generator;
        switch (element)
        {
            case Tetra: case Tetra10: case Tetra20:
                if(biased) generator = new TetraBeamGenerator(cells, size);
                else generator = new HexaBeamGenerator(cells, size);
            break;
            case Pyramid: generator = new PyramidBeamGenerator(cells, size); break;
            case Prism: generator = new PrismBeamGenerator(cells, size); break;
            case Hexa: case Hexa27: generator = new HexaBeamGenerator(cells, size); break;
            default: generator = new TetraBeamGenerator(cells, size); break;
        }
        generator->setTransform(glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = generator->build();
        if(get_linear_element(element) == Tetra && !biased)
            beam_hexa_to_tetra5(mesh->geometry(), mesh->topologies());
        mesh->set_dynamic_topology(false);
        delete generator;
        return mesh;
    }

    Mesh* build_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, Element element, bool biased) {
        Mesh* mesh = get_beam_mesh(pos, cells, size, element, biased);
        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
        if (element == Hexa27) hexa_to_hexa27(mesh->geometry(), mesh->topologies());
        mesh->set_dynamic_geometry(true);
        std::cout << "NB VERTICES " << mesh->nb_vertices() << std::endl;
        return mesh;
    }

    GL_Graphic* build_graphic(const Color& color) {
        return new GL_GraphicHighOrder(2, color);
        //return new GL_GraphicSurface(color);
    }

    GL_DisplayMesh* build_display() {
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->surface() = true;
        display->wireframe() = true;
        display->point() = false;
        return display;
    }

    void add_constraint(Entity* e, const Vector3& pos, const Vector3& size, const SimulationArgs& args, bool gpu) {
        // constraint for Particle system
        if(gpu)
        {
            if(args.scenario_1!=-1)
            {
                const auto rd_constraint_1 = new Cuda_Constraint_Rigid_Controller(new Plane(pos - size + args.dir*0.01f, -args.dir), -args.dir, args.scenario_1);
                //const auto rd_constraint_1 = new Cuda_Constraint_Rigid_Controller(new Box(Vector3(-0.5,-0.2,-2),Vector3(3,0,2)), -args.dir, args.scenario_1);
                //const auto rd_constraint_1 = new Cuda_Constraint_Rigid_Controller(new Sphere(Vector3(0,0.75,0),0.1), -args.dir, args.scenario_1);
                rd_constraint_1->_event_rate = 0.5;
                rd_constraint_1->_smooth_iterations = 30;
                rd_constraint_1->_move_speed = 0.5;
                rd_constraint_1->_rot_speed = 0;
                e->add_component(rd_constraint_1);
            }
            if(args.scenario_2!=-1)
            {
                const auto rd_constraint_2 = new Cuda_Constraint_Rigid_Controller(new Plane(pos + size - args.dir * 0.01f, args.dir), args.dir, args.scenario_2 );
                //const auto rd_constraint_2 = new Cuda_Constraint_Rigid_Controller(new Box(Vector3(-3,-0.2,-2),Vector3(-1,0,2)), -args.dir, args.scenario_1);
                rd_constraint_2->_event_rate = 0.5;
                rd_constraint_2->_smooth_iterations = 30;
                rd_constraint_2->_move_speed = 0.5;
                rd_constraint_2->_rot_speed = 90;
                e->add_component(rd_constraint_2);
            }
        }
        else
        {
            if (args.scenario_1 != -1) {
                const auto rd_constraint_1 = new Constraint_Rigid_Controller(Unit3D::Zero() + args.dir*0.01f, -args.dir, args.scenario_1);
                e->add_component(rd_constraint_1);
                rd_constraint_1->_rot_speed = 90;
                rd_constraint_1->_move_speed = 1.;
            }

            if (args.scenario_2 != -1) {
                auto rd_constraint_2 = new Constraint_Rigid_Controller(pos + size, args.dir, args.scenario_2);
                rd_constraint_2->_rot_speed = 90;
                rd_constraint_2->_move_speed = 1.;
                e->add_component(rd_constraint_2);
            }
        }

    }

    void add_fem_base(Entity* e, const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, bool gpu) {
        add_constraint(e, pos, size, args, gpu);
        e->add_component(build_data_recorder(cells, size, element, e->id()));
        e->add_component(new FEM_DataDisplay(args.display, ColorMap::Viridis));
        e->add_component(build_graphic(color));
        e->add_component(build_display());
    }

    DataRecorder* build_data_recorder(const Vector3I& cells, const Vector3& size, const Element element, const int id) {
        const std::string file_name = std::to_string(id) + "_" +  std::string(element_name(element)) + "_" + std::to_string(cells.x) + "_" + std::to_string(cells.y) + "_" + std::to_string(cells.z)
            + "_" + std::to_string(static_cast<int>(size.x)) + "x" + std::to_string(static_cast<int>(size.y)) + "x" + std::to_string(static_cast<int>(size.z));

        DataRecorder* data_recorder = new DataRecorder(file_name, false);
        data_recorder->add(new TimeRecorder());
        data_recorder->add(new MeshRecorder());
        data_recorder->add(new FEM_VTK_Recorder(file_name));
        data_recorder->add(new Graphic_VTK_Recorder(file_name));
        //data_recorder->add(new FEM_Torsion_error_recorder(180,4));
        //data_recorder->add(new Mesh_Diff_VTK_Recorder(file_name, 1));
        //data_recorder->add(new FEM_Flexion_error_recorder(Vector3(4, 1, 1), Vector3(3.34483, -1.86949, 1.0009)));
        return data_recorder;
    }

    Mesh* build_vtk_mesh(const Vector3& pos, const std::string& file, const Element element, const SimulationArgs& args) {
        VTK_Loader loader(AppInfo::PathToAssets() + file);
        loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = loader.build();
        mesh_post_process(mesh, element, args);
        return mesh;
    }

    Mesh* build_msh_mesh(const Vector3& pos, const std::string& file, const Element element, const SimulationArgs& args) {
        Msh_Loader loader(AppInfo::PathToAssets() + file);
        loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = loader.build();
        mesh_post_process(mesh, element, args);
        return mesh;
    }

    void mesh_post_process(Mesh* mesh, const Element element, const SimulationArgs& args) {
        if(args.refine > 0) {
            const Element lin_elem = get_linear_element(element);
            MeshMap* map = lin_elem == Tetra ? tetra_to_linear(mesh, Tetra, args.refine) : hexa_to_linear(mesh, element, args.refine);
            map->apply_to_mesh(mesh);
            if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
            if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
            if (element == Hexa27) hexa_to_hexa27(mesh->geometry(), mesh->topologies());
        }
        else {
            if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
            if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
            if (element == Hexa27) hexa_to_hexa27(mesh->geometry(), mesh->topologies());
        }
        mesh->set_dynamic_geometry(true);
    }


    void add_fem_mesh(Entity* e, const Vector3& pos, const Vector3I& cells, const Vector3& size, const Element element, const SimulationArgs& args) {
        if(args.mesh_type.empty()) e->add_behaviour(build_beam_mesh(pos, cells, size, element, args.biased));
        else if(args.mesh_type == "vtk") e->add_behaviour(build_vtk_mesh(pos, args.mesh_file, element, args));
        else e->add_behaviour(build_msh_mesh(pos, args.mesh_file, element, args));
    }

    void build_mesh_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args) {
        Entity* e = Engine::CreateEnity();
        add_fem_mesh(e, pos, cells, size, element, args);
        if(e->id() == 1 && args.mesh_type == "vtk") e->add_component(new VTK_Attribute(args.mesh_file, "u"));
        e->add_component(build_graphic(color));
        e->add_component(build_display());
    }

    void build_fem_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, bool gpu) {
        Entity* e = Engine::CreateEnity();
        add_fem_mesh(e, pos, cells, size, element, args);
        if(gpu) e->add_component(new Cuda_FEM_Dynamic(std::max(args.iteration, args.sub_iteration), args.density, args.distribution, args.young, args.poisson, args.material, args.damping));
        else e->add_component(new FEM_Dynamic_Generic(args.density, args.distribution, args.young, args.poisson, args.material, std::max(args.iteration, args.sub_iteration)));
        add_fem_base(e, pos, cells, size, color, element, args, gpu);
    }

    void build_mg_vbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, const scalar rho, const scalar linear, const bool gpu) {
        Entity* e = Engine::CreateEnity();
        add_fem_mesh(e, pos, cells, size, element, args);
        if(gpu) e->add_component(new Cuda_MG_VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material, args.iteration, args.sub_iteration, args.damping, rho, linear));
        else e->add_component(new MG_VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material,args.iteration, args.sub_iteration, args.damping, rho, linear));
        add_fem_base(e, pos, cells, size, color, element, args, gpu);
    }

    void build_vbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, const float rho, bool gpu) {
        Entity* e = Engine::CreateEnity();
        add_fem_mesh(e, pos, cells, size, element, args);
        if(gpu) e->add_component(new Cuda_VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material, args.iteration, args.sub_iteration, args.damping, rho));
        else e->add_component(new VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material,args.iteration, args.sub_iteration, args.damping, rho));
        add_fem_base(e, pos, cells, size, color, element, args, gpu);
    }

    void build_mixed_vbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, const int exp_it) {
        Entity* e = Engine::CreateEnity();
        add_fem_mesh(e, pos, cells, size, element, args);
        e->add_behaviour(build_beam_mesh(pos, cells, size, element, args.biased));
        e->add_component(new Cuda_Mixed_VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material, args.iteration, args.sub_iteration, exp_it, args.damping));
        add_fem_base(e, pos, cells, size, color, element, args, true);
    }

    void build_lf_vbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, const scalar rho) {
        Entity* e = Engine::CreateEnity();
        add_fem_mesh(e, pos, cells, size, element, args);
        e->add_behaviour(build_beam_mesh(pos, cells, size, element, args.biased));
        e->add_component(new Cuda_LF_VBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson, args.material, args.iteration, args.sub_iteration,  args.damping, rho));
        add_fem_base(e, pos, cells, size, color, element, args, true);
    }

    void build_xpbd_entity(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, const Element element, const SimulationArgs& args, const bool gpu, const bool coupled) {
        Entity* e = Engine::CreateEnity();
        add_fem_mesh(e, pos, cells, size, element, args);
        if(gpu) e->add_component(new Cuda_XPBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson,args.material,std::max(args.iteration, args.sub_iteration), args.damping, coupled));
        else e->add_component(new XPBD_FEM_Dynamic(args.density, args.distribution, args.young, args.poisson,args.material, args.iteration, args.sub_iteration, args.damping, coupled));
        add_fem_base(e, pos, cells, size, color, element, args, gpu);
    }


};
