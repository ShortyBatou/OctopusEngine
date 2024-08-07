#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"

#include "Manager/OpenglManager.h"
#include "Manager/TimeManager.h"
#include "Manager/CameraManager.h"


#include "Rendering/GL_GraphicElement.h"
#include "Rendering/GL_GraphicSurface.h"
#include "Rendering/GL_DisplayMode.h"

#include "UI/UI_Primitive.h"

#include "Scene/SceneManager.h"

#include "Mesh/Mesh.h"
#include "Script/Record/DataRecorder.h"
#include "Script/Dynamic/XPBD_FEM_Dynamic.h"
#include "Script/Dynamic/Constraint_Rigid_Controller.h"
#include "Script/Display/DataDisplay.h"
class UI_SceneManager : public UI_Display {
public:
	UI_SceneManager() : _item_current(SceneManager::SceneID()) 
	{ 
		std::cout << "UI SCENE " << _item_current << std::endl;
	}

	virtual char* name() override {
		return "Scene";
	}

	virtual void draw() override {
		if (ImGui::CollapsingHeader(name(), ImGuiTreeNodeFlags_DefaultOpen))
		{
			std::vector<char*> scene_names;

			for (Scene* ui : SceneManager::Scenes()) {
				scene_names.push_back(ui->name());
			}
			ImGui::Text("Select : ");
			ImGui::SameLine();
			ImGui::Combo("##scene_combo", &_item_current, scene_names.data(), scene_names.size());
			ImGui::SameLine();
			ImGui::Spacing();
			if (ImGui::Button("Build")) {
				SceneManager::SetScene(_item_current);
			}
		}
	}
protected:
	int _item_current;
};

class UI_DisplaySettings : public UI_Component {
public:
	virtual char* name() override {
		return "Colors";
	}

	bool can_draw(Entity* entity) override {
		OpenGLManager* opengl_manager = entity->get_component<OpenGLManager>();
		if (!opengl_manager) return false;
		return true;
	}

	virtual void draw(Entity* entity) override {
		OpenGLManager* opengl_manager = entity->get_component<OpenGLManager>();;

		ImGui::ColorEdit3("Background Color", &opengl_manager->background().x);
		ImGui::SliderFloat("Wireframe Intencity", &GL_Graphic::wireframe_intencity, 0.0f, 1.0f, "ratio = %.3f");
		ImGui::ColorEdit3("Vertices Color", &GL_Graphic::vertice_color.x);
		ImGui::SeparatorText("Element's Color");

		ImGui::SliderFloat("Vertices Size", &GL_Graphic::vertice_size, 1.0f, 20.0f, "ratio = %.5f");
		ImGui::SliderFloat("Line Size", &GL_Graphic::line_size, 1.0f, 10.0f, "ratio = %.5f");

		//bool change = false;
		//change = change + ImGui::ColorEdit3("Line", &GL_GraphicElement::element_colors[Line].x);
		//change = change + ImGui::ColorEdit3("Triangle", &GL_GraphicElement::element_colors[Triangle].x);
		//change = change + ImGui::ColorEdit3("Quad", &GL_GraphicElement::element_colors[Quad].x);
		//change = change + ImGui::ColorEdit3("Tetra", &GL_GraphicElement::element_colors[Tetra].x);
		//change = change + ImGui::ColorEdit3("Pyramid", &GL_GraphicElement::element_colors[Pyramid].x);
		//change = change + ImGui::ColorEdit3("Prysm", &GL_GraphicElement::element_colors[Prism].x);
		//change = change + ImGui::ColorEdit3("Hexa", &GL_GraphicElement::element_colors[Hexa].x);
		//change = change + ImGui::ColorEdit3("Tetra10", &GL_GraphicElement::element_colors[Tetra10].x);
		//if (change) {
		//	for (Entity* e : Engine::GetEntities()) {
		//		GL_GraphicElement* gl_elements = e->get_component<GL_GraphicElement>();
		//		if (gl_elements) {
		//			gl_elements->update_buffer_colors();
		//		}
		//	}
		//}
	}
protected:

};

class UI_Time : public UI_Component {
public:
	virtual char* name() override {
		return "Time";
	}
	virtual void init() {
		_fixed_delta_t = Time::Fixed_DeltaTime();
	}

	bool can_draw(Entity* entity) override {
		return true;
	}

	virtual void draw(Entity* entity) override {
		ImGui::Text("Time = %.1f s", Time::Timer());
		ImGui::Text("Fixed Time = %.1f s", Time::Fixed_Timer()); 
		ImGui::Text("Frames = %d", Time::Frame());

		ImGui::Text("dt = %.2f ms  -  %.1f FPS", Time::DeltaTime() * 1000.0f, 1. / Time::DeltaTime());
		if (ImGui::DragFloat("Fixed Delta Time", &_fixed_delta_t, 0.00001f, 0.00001f, 1.0f, "%.06f s")) {
			Time::Instance().set_fixed_deltaTime(_fixed_delta_t);
			std::cout << Time::Fixed_DeltaTime() << std::endl;
		}
	}
protected:
	scalar _fixed_delta_t;
};

class UI_Dynamic : public UI_Component {
public:
	virtual char* name() override {
		return "Dynamic";
	}
	virtual void init() {
		_gravity = Dynamic::gravity();
	}

	bool can_draw(Entity* entity) override {
		return true;
	}

	virtual void draw(Entity* entity) override {
		if (ImGui::InputFloat3("gravity ", &_gravity.x)) {
			Dynamic::Instance().set_gravity(_gravity);
		}
		
	}
protected:
	Vector3 _gravity;
};


class UI_Camera : public UI_Component {
public:
	virtual char* name() override {
		return "Camera";
	}

	virtual void init() {
	}

	bool can_draw(Entity* entity) override {
		return entity->get_component<CameraManager>() != nullptr;
	}

	virtual void draw(Entity* entity) override {
		CameraManager* camera_manager = entity->get_component<CameraManager>();
		ImGui::SliderFloat("Speed", &camera_manager->speed(), 0.0f, 1.0f, "ratio = %.3f");
		ImGui::SliderFloat("Zoom", &camera_manager->zoom(), camera_manager->zoom_range().x, camera_manager->zoom_range().y, "ratio = %.3f");
	}
protected:
};



class UI_Mesh_Display : public UI_Component {
public:


	virtual char* name() override {
		return "Mesh Display";
	}


	bool can_draw(Entity* entity) override {
		GL_DisplayMesh* gl_display = entity->get_component<GL_DisplayMesh>();
		if (gl_display) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		GL_Graphic* gl_graphic = entity->get_component< GL_Graphic>();
		GL_DisplayMesh* gl_display = entity->get_component<GL_DisplayMesh>();
		static const char* modes[] = { "Point", "Surface", "Surface High-Order", "Element" };
		if (ImGui::Combo("Color Map", &current_mode, modes, 4)) {
			GL_Graphic* new_graphic = nullptr;
			gl_display->surface() = true;
			switch (current_mode)
			{
				case 0: new_graphic = new GL_Graphic(gl_graphic->color()); gl_display->point() = true; gl_display->surface() = false;  gl_display->wireframe() = false; break;
				case 1: new_graphic = new GL_GraphicSurface(gl_graphic->color()); break;
				case 2: new_graphic = new GL_GraphicHighOrder(2, gl_graphic->color()); break;
				case 3: new_graphic = new GL_GraphicElement(gl_graphic->color()); break;
			}
			std::cout << gl_graphic->color().r << " " << gl_graphic->color().g << " " << gl_graphic->color().b << std::endl;
			Mesh* mesh = entity->get_component<Mesh>();
			mesh->update_mesh();
			new_graphic->color() = gl_graphic->color();
			entity->remove_component(gl_graphic);
			entity->add_component(new_graphic);
			new_graphic->init();
			new_graphic->late_init();
			
			gl_display->set_graphic(new_graphic);
			gl_graphic = new_graphic;
		}

		if (current_mode == 3) {
			GL_GraphicElement* gl_graphic_element = entity->get_component< GL_GraphicElement>();
			ImGui::SliderFloat("Element Scale", &gl_graphic_element->scale(), 0.0f, 1.0f, "ratio = %.05f");
		}
		if (gl_graphic) {
			ImGui::ColorEdit3("Color", &gl_graphic->color().r);
		}

		if (gl_display) {
			ImGui::Checkbox("Wireframe", &gl_display->wireframe());
			ImGui::SameLine();
			ImGui::Checkbox("Surface", &gl_display->surface());
			ImGui::SameLine();
			ImGui::Checkbox("Point", &gl_display->point());
		}

	}

private:
	int current_mode;
};


class UI_Data_Recorder : public UI_Component {
public:
	UI_Data_Recorder() : UI_Component() {
		saved = false;
	}

	virtual char* name() override {
		return "Data Recorder";
	}


	bool can_draw(Entity* entity) override {
		DataRecorder* data_recorder = entity->get_component<DataRecorder>();
		if (data_recorder) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		DataRecorder* data_recorder = entity->get_component<DataRecorder>();
		ImGui::Text(("File : " + data_recorder->json_path()).c_str());
		if (ImGui::Button("Save")) {
			data_recorder->save();
			saved = true;
			save_frame = Time::Frame();
		}
		if (saved) {
			ImGui::SameLine();
			ImGui::Text("Last save at frame %d", save_frame);
		}
	}
protected: 
	bool saved;
	int save_frame;
};

class UI_Data_Displayer : public UI_Component {
public:
	UI_Data_Displayer() : UI_Component() {
		for (int d = FEM_DataDisplay::Type::BaseColor; d != FEM_DataDisplay::Type::None; ++d) {
			if (d == FEM_DataDisplay::Type::None) break;
			str_display_types.push_back(FEM_DataDisplay::Type_To_Str(FEM_DataDisplay::Type(d)));
		}
		for (int d = ColorMap::Type::Default; d != ColorMap::Type::BnW + 1; ++d) {
			str_colormap_types.push_back(ColorMap::Type_To_Str(ColorMap::Type(d)));
		}
	}

	virtual char* name() override {
		return "FEM Data Displayer";
	}


	bool can_draw(Entity* entity) override {
		FEM_DataDisplay* data_display = entity->get_component<FEM_DataDisplay>();
		if (data_display) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		FEM_DataDisplay* data_recorder = entity->get_component<FEM_DataDisplay>();
		int _current_display = data_recorder->_mode;
		int _current_colormap = data_recorder->color_map;
		if (ImGui::Combo("Data Type", &_current_display, str_display_types.data(), str_display_types.size())) {
			data_recorder->_mode = FEM_DataDisplay::Type(_current_display);
			//data_recorder->update();
		}
		if (ImGui::Combo("Color Map", &_current_colormap, str_colormap_types.data(), str_colormap_types.size())) {
			data_recorder->color_map = ColorMap::Type(_current_colormap);
			//data_recorder->update();
		}
	}
protected:
	
	std::vector<char*> str_colormap_types;
	std::vector<char*> str_display_types;
};

class UI_Graphic_Saver : public UI_Component {
public:
	UI_Graphic_Saver() : UI_Component() {
		saved = false;
	}

	virtual char* name() override {
		return "Graphic Mesh Saver";
	}


	bool can_draw(Entity* entity) override {
		VTK_Graphic* vtk_fem = entity->get_component<VTK_Graphic>();
		if (vtk_fem) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		VTK_Graphic* vtk_fem = entity->get_component<VTK_Graphic>();
		ImGui::Text(("File : " + vtk_fem->file_name()).c_str());
		if (ImGui::Button("Save")) {
			vtk_fem->save();
			saved = true;
			save_frame = Time::Frame();
		}
		if (saved) {
			ImGui::SameLine();
			ImGui::Text("Last save at frame %d", save_frame);
		}
	}
protected:
	bool saved;
	int save_frame;
};


class UI_PBD_Dynamic : public UI_Component {
public:
	UI_PBD_Dynamic() : UI_Component() {
	}

	virtual char* name() override {
		return "PBD";
	}


	bool can_draw(Entity* entity) override {
		XPBD_FEM_Dynamic* pbd = entity->get_component<XPBD_FEM_Dynamic>();
		if (pbd) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		XPBD_FEM_Dynamic* pbd = entity->get_component<XPBD_FEM_Dynamic>();
		it = pbd->get_iteration();
		sub_it = pbd->get_sub_iteration();
		if (ImGui::InputInt("steps", &it) || ImGui::InputInt("sub steps", &sub_it)) {
			it = std::max(0, it);
			sub_it = std::max(0, sub_it);
			pbd->set_iterations(it, sub_it);
		}
	}
protected:
	int it, sub_it;
};


class UI_Constraint_Rigid_Controller : public UI_Component {
public:
	UI_Constraint_Rigid_Controller() : UI_Component() { }

	virtual char* name() override {
		return "Constraint Rigid";
	}


	bool can_draw(Entity* entity) override {
		Constraint_Rigid_Controller* rc = entity->get_component<Constraint_Rigid_Controller>();
		if (rc) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		std::vector<Constraint_Rigid_Controller*> components = entity->get_components<Constraint_Rigid_Controller>();
		if (components.size() == 0) return;
		
		Constraint_Rigid_Controller* rc = components[0];
		if (ImGui::InputInt("Mode", &rc->_mode)
			|| ImGui::InputFloat("Event Rate", &rc->_event_rate)
			|| ImGui::SliderInt("Smoothing Itetrations", &rc->_smooth_iterations, 1, 30)
			|| ImGui::InputFloat("Move Speed", &rc->_move_speed)
			|| ImGui::InputFloat("Rotation Speed", &rc->_rot_speed))
		{
			for (unsigned int i = 1; i < components.size(); ++i) {
				components[i]->_mode = rc->_mode;
				components[i]->_move_speed = rc->_move_speed;
				components[i]->_rot_speed = rc->_rot_speed;
				components[i]->_smooth_iterations = rc->_smooth_iterations;
				components[i]->_event_rate = rc->_event_rate;
			}
		}
	}
};