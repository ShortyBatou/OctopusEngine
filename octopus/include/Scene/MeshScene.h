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
        editor->add_manager_ui(new UI_SceneColor());
        editor->add_manager_ui(new UI_Camera());
        editor->add_component_ui(new UI_Mesh());
        editor->add_component_ui(new UI_Data_Recorder());
        editor->add_component_ui(new UI_Graphic_Saver());
        editor->add_component_ui(new UI_PBD_Dynamic());
        editor->add_component_ui(new UI_Constraint_Rigid_Controller());
    }

    virtual void build_root(Entity* root) override
    {
        root->addBehaviour(new TimeManager(1. / 60.));
        root->addBehaviour(new InputManager());
        root->addBehaviour(new CameraManager());
        root->addBehaviour(new DebugManager(true));
        root->addBehaviour(new OpenGLManager(Color(0.9, 0.9, 0.9, 1.)));
    }

    // build scene's entities
    virtual void build_entities() override
    {
        Vector3 size(1, 1, 1);
        Vector3I cells;

        cells = Vector3I(1, 1, 1);
        build_beam_mesh(Vector3(0, 0, 0), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra10);
        //build_xpbd_entity(Vector3(0, 0, 0), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra, false, false);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.3, 0.3, 0.8, 1.), Tetra, false, false);
        //cells = Vector3I(8, 3, 3);
        //build_vtk_mesh(Vector3(0, 0, 0), cells, size, Color(0.3, 0.3, 0.8, 1.), "result/vtk/Torsion/Torsion_Bad_MeshTetra10_8_2_2.vtk");
        //cells = Vector3I(6, 2, 2);
        //build_xpbd_entity(Vector3(0, 0, 2), cells, size, Color(0.8, 0.3, 0.3, 1.), Tetra, false, true);
        //build_xpbd_entity(Vector3(0, 0, 1), cells, size, Color(0.8, 0.3, 0.8, 1.), Tetra10, true, false);
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

    void tetra10_refinement(Mesh::Geometry& geometry, std::map<Element, Mesh::Topology>& topologies) {
        Mesh::Topology tetras = topologies[Tetra10];

        TetraConverter* tetra_converter = new TetraConverter();
        tetra_converter->init();
        Mesh::Topology ref_tetra_edges = tetra_converter->get_elem_topo_edges();
        Mesh::Geometry ref_tetra_geom = tetra_converter->geo_ref();

        unsigned int nb_tetra = tetras.size() / 10 * 4;

        // rebuild the mesh as linear tetrahedron mesh but with only position in reference element
        std::vector<unsigned int> v_ids(geometry.size(), -1); // permit to check if vertices allready defined or not
        std::vector<unsigned int> t_ids(nb_tetra); // in which tetrahedron is defined each tetrahedron t_id = [0,nb_tetra-1]
        std::vector<int> v_tetra; // in which element the vertices is valid
        Mesh::Geometry ref_geometry; // vertices position in reference element
        Mesh::Geometry ref_tetra_geometry(nb_tetra); // vertices position of all linear tetra (in ref element)
        Mesh::Topology tetra_topology(nb_tetra); // topology of linear tetra
        unsigned int v_id = 0;
        unsigned int t_id = 0;
        for (unsigned int i = 0; i < tetras.size(); i += 10) {
            t_id = i / 10;
            t_ids[t_id] = t_id;
            for (unsigned int j = 0; j < 4; ++j) // we only needs the first 4 vertices
            {
                unsigned int k = t_id * 4 + j;
                ref_tetra_geometry[k] = ref_tetra_geom[j];
                unsigned int id = tetras[i + j];
                if (v_ids[id] == -1) {
                    v_tetra.push_back(t_id);
                    ref_geometry.push_back(ref_tetra_geom[j]);
                    tetra_topology[k] = v_id;

                    v_ids[id] = v_id;
                    v_id++;
                }
                else {
                    tetra_topology[i / 10 * 4 + j] = v_ids[id];
                }
            }
        }
        
        

        unsigned int tetra_10_topo[32] = { 0,4,6,7, 1,5,4,8, 7,8,9,3, 2,6,5,9, 6,4,5,7, 7,4,5,8, 6,5,9,7, 7,8,5,9 };
        std::map<Face<2>, unsigned int> edges;
        Mesh::Topology new_tetra_topology;
        std::vector<int> new_v_tetra;
        Mesh::Topology e_topo(2);
        std::vector<unsigned int> ids(10);
        for (unsigned int i = 4; i < 8; i += 4) {
            t_id = t_ids[i / 4];
            for (unsigned int j = 0; j < 4; ++j) ids[j] = tetra_topology[i + j];

            for (unsigned int j = 0; j < ref_tetra_edges.size(); j+=2) {
                e_topo[0] = tetra_topology[i + ref_tetra_edges[j]]; 
                e_topo[1] = tetra_topology[i + ref_tetra_edges[j+1]];
                Face<2> e(e_topo);
                unsigned int id;
                // edge found in map
                if (edges.find(e) != edges.end()) {
                    id = edges[e];
                }
                else {
                    id = ref_geometry.size();
                    Vector3 pa = ref_tetra_geometry[e_topo[0]];
                    Vector3 pb = ref_tetra_geometry[e_topo[1]];

                    Vector3 p = scalar(0.5) * (pa + pb);
                    ref_geometry.push_back(p);
                    v_tetra.push_back(t_id);

                    edges[e] = id;
                }
                ids[4+j/2] = id;
            }

            for (unsigned int k = 0; k < 32; ++k) {
                new_tetra_topology.push_back(ids[tetra_10_topo[k]]);
            }

            for (unsigned int k = 0; k < 8; ++k) {
                new_v_tetra.push_back(t_id);
            }

        }


        Tetra_10* shape = new Tetra_10();
        topologies[Tetra] = new_tetra_topology;
        Mesh::Geometry new_geometry(ref_geometry.size());
        for (unsigned int i = 0; i < ref_geometry.size(); ++i) {
            t_id = v_tetra[i];
            new_geometry[i] = Vector3(0., 0., 0.);
            Vector3 p = ref_geometry[i];
            std::vector<scalar> weights = shape->build_shape(p.x, p.y, p.z);
            for (unsigned int j = 0; j < weights.size(); ++j) {
                new_geometry[i] += geometry[tetras[t_id * 10 + j]] * weights[j];
            }
        }
        geometry = new_geometry;
        topologies[Tetra10].clear();
        delete shape;


        std::cout << "DAM" << std::endl;

    }

    void build_vtk_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, std::string file) {
        VTK_Loader loader(AppInfo::PathToAssets() + file);
        loader.setTransform(glm::scale(Vector3(1.f)) * glm::translate(Matrix::Identity4x4(), pos));
        Mesh* mesh = loader.build();
        std::vector<Vector3> u_v3 = loader.get_point_data_v3("u");
        tetra10_refinement(mesh->geometry(), mesh->topologies());

        build_entity(mesh, color);
    }

    void build_beam_mesh(const Vector3& pos, const Vector3I& cells, const Vector3& size, const Color& color, Element element) {
        Mesh* mesh;
        mesh = get_beam_mesh(pos, cells, size, element);
        if (element == Tetra10) tetra4_to_tetra10(mesh->geometry(), mesh->topologies());
        if (element == Tetra20) tetra4_to_tetra20(mesh->geometry(), mesh->topologies());
        mesh->set_dynamic_geometry(true);
        tetra10_refinement(mesh->geometry(), mesh->topologies());
        build_entity(mesh, color);
    }

    void build_entity(Mesh* mesh, const Color& color) {
        Entity* e = Engine::CreateEnity();
        e->addBehaviour(mesh);

        // Mesh converter simulation to rendering (how it will be displayed)
        GL_Graphic* graphic;
        int type = (mesh->topologies()[Tetra10].size() != 0 || mesh->topologies()[Tetra20].size() != 0) ? 3 : 1;
        //graphic = new GL_GraphicElement(0.7);
        switch (type) {
            case 0 : graphic = new GL_GraphicSurface(color);
            case 1 : graphic = new GL_GraphicElement(0.7);
            case 2 : graphic = new GL_GraphicHighOrder(3, color);
            default: graphic = new GL_GraphicSurface(color);
        }

        graphic->normals() = false;
        e->addComponent(graphic);

        // Opengl Rendering
        GL_DisplayMesh* display = new GL_DisplayMesh();
        display->wireframe() = true;
        display->point() = false;
        display->normal() = false;
        e->addComponent(display);

        // save mesh in VTK format (Paraview)

        std::string file_name = "Mesh";
        DataRecorder* data_recorder = new DataRecorder(file_name);
        data_recorder->add(new MeshRecorder());
        data_recorder->add(new Graphic_VTK_Recorder(file_name));
        e->addComponent(data_recorder);
    }
};