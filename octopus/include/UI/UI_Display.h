#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Manager/OpenglManager.h"
#include "Manager/TimeManager.h"
#include "Rendering/GL_GraphicElement.h"
#include "Mesh/Mesh.h"
class UI_Display : public Component {
public:
	UI_Display(Entity* entity) : Component(entity) { }
	virtual void draw() = 0;
	virtual char* name() = 0;
};

class UI_SceneColor : public UI_Display {
public:
	UI_SceneColor(Entity* entity) : UI_Display(entity) {
		_opengl_manager = entity->getComponent<OpenGLManager>();
		assert(_opengl_manager != nullptr);
	}
	virtual char* name() override {
		return "Colors";
	}
	virtual void draw() override {
		ImGui::ColorEdit3("Background Color", &_opengl_manager->background().x);
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
	OpenGLManager* _opengl_manager;
};

class UI_Time : public UI_Display {
public:
	UI_Time(Entity* entity) : UI_Display(entity) {
		assert(entity->getComponent<TimeManager>() != nullptr);
		_fixed_delta_t = Time::Fixed_DeltaTime();
	}
	virtual char* name() override {
		return "Time";
	}
	virtual void update() {
		
	}
	virtual void draw() override {
		ImGui::Text("Time = %.1f s", Time::Timer());
		ImGui::Text("dt = %.2f ms  -  %.1f FPS", Time::DeltaTime() * 1000.0f, 1./Time::DeltaTime());
		if (ImGui::DragFloat("Fixed Delta Time", &_fixed_delta_t, 0.00001f, 0.00001f, 1.0f, "%.06f s")) {
			Time::Instance().set_fixed_deltaTime(_fixed_delta_t);
		}
	}
protected:
	scalar _fixed_delta_t;
};