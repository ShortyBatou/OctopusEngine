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
#include "Script/VTK/VTK_FEM.h"

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
		change = change + ImGui::ColorEdit3("Prysm", &GL_GraphicElement::element_colors[Prysm].x);
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

class UI_FEM_Saver : public UI_Component {
public:
	UI_FEM_Saver() : UI_Component() {
		saved = false;
	}

	virtual char* name() override {
		return "FEM Mesh Saver";
	}


	bool can_draw(Entity* entity) override {
		VTK_FEM* vtk_fem = entity->getComponent<VTK_FEM>();
		if (vtk_fem) return true;
		return false;
	}

	virtual void draw(Entity* entity) override {
		VTK_FEM* vtk_fem = entity->getComponent<VTK_FEM>();
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