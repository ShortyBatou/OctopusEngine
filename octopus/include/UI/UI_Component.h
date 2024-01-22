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

class UI_SceneColor : public UI_Component {
public:
	virtual char* name() override {
		return "Colors";
	}

	bool can_draw(Entity* entity) override {
		OpenGLManager* opengl_manager = entity->getComponent<OpenGLManager>();
		if (!opengl_manager) return false;
		return true;
	}

	virtual void draw(Entity* entity) override {
		OpenGLManager* opengl_manager = entity->getComponent<OpenGLManager>();;

		ImGui::ColorEdit3("Background Color", &opengl_manager->background().x);
		ImGui::SliderFloat("Wireframe Intencity", &GL_Graphic::wireframe_intencity, 0.0f, 1.0f, "ratio = %.3f");
		ImGui::ColorEdit3("Vertices Color", &GL_Graphic::vertice_color.x);
		ImGui::SeparatorText("Element's Color");
		bool change = false;
		change = change + ImGui::ColorEdit3("Line", &GL_GraphicElement::element_colors[Line].x);
		change = change + ImGui::ColorEdit3("Triangle", &GL_GraphicElement::element_colors[Triangle].x);
		change = change + ImGui::ColorEdit3("Quad", &GL_GraphicElement::element_colors[Quad].x);
		change = change + ImGui::ColorEdit3("Tetra", &GL_GraphicElement::element_colors[Tetra].x);
		change = change + ImGui::ColorEdit3("Pyramid", &GL_GraphicElement::element_colors[Pyramid].x);
		change = change + ImGui::ColorEdit3("Prysm", &GL_GraphicElement::element_colors[Prism].x);
		change = change + ImGui::ColorEdit3("Hexa", &GL_GraphicElement::element_colors[Hexa].x);
		change = change + ImGui::ColorEdit3("Tetra10", &GL_GraphicElement::element_colors[Tetra10].x);
		if (change) {
			for (Entity* e : Engine::GetEntities()) {
				GL_GraphicElement* gl_elements = e->getComponent<GL_GraphicElement>();
				if (gl_elements) {
					gl_elements->update_buffer_colors();
				}
			}
		}
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


class UI_Camera : public UI_Component {
public:
	virtual char* name() override {
		return "Camera";
	}

	virtual void init() {
	}

	bool can_draw(Entity* entity) override {
		return entity->getComponent<CameraManager>() != nullptr;
	}

	virtual void draw(Entity* entity) override {
		CameraManager* camera_manager = entity->getComponent<CameraManager>();
		ImGui::SliderFloat("Speed", &camera_manager->speed(), 0.0f, 1.0f, "ratio = %.3f");
		ImGui::SliderFloat("Zoom", &camera_manager->zoom(), camera_manager->zoom_range().x, camera_manager->zoom_range().y, "ratio = %.3f");
	}
protected:
};



class UI_Mesh : public UI_Component {
public:
	virtual char* name() override {
		return "Mesh";
	}


	bool can_draw(Entity* entity) override {
		GL_GraphicSurface* gl_surface = entity->getComponent<GL_GraphicSurface>();
		GL_DisplayMesh* gl_display = entity->getComponent<GL_DisplayMesh>();
		if (gl_surface || gl_display) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		GL_GraphicSurface* gl_surface = entity->getComponent< GL_GraphicSurface>();
		if (gl_surface) {
			ImGui::ColorEdit3("Color", &gl_surface->color().r);
		}

		GL_DisplayMesh* gl_display = entity->getComponent<GL_DisplayMesh>();
		if (gl_display) {
			ImGui::Checkbox("Wireframe", &gl_display->wireframe());
			ImGui::SameLine();
			ImGui::Checkbox("Surface", &gl_display->surface());
			ImGui::SameLine();
			ImGui::Checkbox("Point", &gl_display->point());
			ImGui::SameLine();
			ImGui::Checkbox("Normal", &gl_display->normal());
			ImGui::ColorEdit3("Normal Color", &gl_display->normal_color().r);
			ImGui::SliderFloat("Normal Length", &gl_display->normal_length(), 0.0f, 1.0f, "ratio = %.3f");
		}

	}
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
		DataRecorder* data_recorder = entity->getComponent<DataRecorder>();
		if (data_recorder) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		DataRecorder* data_recorder = entity->getComponent<DataRecorder>();
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

class UI_Graphic_Saver : public UI_Component {
public:
	UI_Graphic_Saver() : UI_Component() {
		saved = false;
	}

	virtual char* name() override {
		return "Graphic Mesh Saver";
	}


	bool can_draw(Entity* entity) override {
		VTK_Graphic* vtk_fem = entity->getComponent<VTK_Graphic>();
		if (vtk_fem) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		VTK_Graphic* vtk_fem = entity->getComponent<VTK_Graphic>();
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
		XPBD_FEM_Dynamic* pbd = entity->getComponent<XPBD_FEM_Dynamic>();
		if (pbd) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		XPBD_FEM_Dynamic* pbd = entity->getComponent<XPBD_FEM_Dynamic>();
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
		Constraint_Rigid_Controller* rc = entity->getComponent<Constraint_Rigid_Controller>();
		if (rc) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		std::vector<Constraint_Rigid_Controller*> components = entity->getComponents<Constraint_Rigid_Controller>();
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