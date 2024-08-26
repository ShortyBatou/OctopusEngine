#include "UI/UI_Component.h"
#include <Rendering/GL_GraphicHighOrder.h>
#include "imgui.h"

void UI_SceneManager::draw() {
	if (ImGui::CollapsingHeader(name().c_str(), ImGuiTreeNodeFlags_DefaultOpen))
	{
		std::vector<char*> scene_names;

		for (Scene* ui : SceneManager::Scenes()) {
			scene_names.push_back(ui->name());
		}
		ImGui::Text("Select : ");
		ImGui::SameLine();
		ImGui::Combo("##scene_combo", &_item_current, scene_names.data(), static_cast<int>(scene_names.size()));
		ImGui::SameLine();
		ImGui::Spacing();
		if (ImGui::Button("Build")) {
			SceneManager::SetScene(_item_current);
		}
	}
}


void UI_DisplaySettings::draw(Entity* entity) {
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

void UI_Time::draw(Entity* entity) {
	ImGui::Text("Time = %.1f s", Time::Timer());
	ImGui::Text("Fixed Time = %.1f s", Time::Fixed_Timer());
	ImGui::Text("Frames = %d", Time::Frame());

	ImGui::Text("dt = %.2f ms  -  %.1f FPS", Time::DeltaTime() * 1000.0f, 1. / Time::DeltaTime());
	if (ImGui::DragFloat("Fixed Delta Time", &_fixed_delta_t, 0.00001f, 0.00001f, 1.0f, "%.06f s")) {
		Time::Instance().set_fixed_deltaTime(_fixed_delta_t);
		std::cout << Time::Fixed_DeltaTime() << std::endl;
	}
}


void UI_Dynamic::draw(Entity* entity) {
	if (ImGui::InputFloat3("gravity ", &_gravity.x)) {
		Dynamic::Instance().set_gravity(_gravity);
	}
}


void UI_Camera::draw(Entity* entity) {
	CameraManager* camera_manager = entity->get_component<CameraManager>();
	ImGui::SliderFloat("Speed", &camera_manager->speed(), 0.0f, 1.0f, "ratio = %.3f");
	ImGui::SliderFloat("Zoom", &camera_manager->zoom(), camera_manager->zoom_range().x, camera_manager->zoom_range().y, "ratio = %.3f");
}



void UI_Mesh_Display::draw(Entity* entity) {

	GL_Graphic* gl_graphic = entity->get_component< GL_Graphic>();
	GL_DisplayMesh* gl_display = entity->get_component<GL_DisplayMesh>();

	const int id = entity->id() - 1;
	if(id >= current_mode.size()) {
		int c = 0;
		if(dynamic_cast<GL_GraphicSurface*>(gl_graphic)) c = 1;
		else if(dynamic_cast<GL_GraphicHighOrder*>(gl_graphic)) c = 2;
		else if(dynamic_cast<GL_GraphicElement*>(gl_graphic)) c = 3;
		current_mode.push_back(c);
	}

	static const char* modes[] = { "Point", "Surface", "Surface High-Order", "Element" };
	if (ImGui::Combo("Color Map", &current_mode[id], modes, 4)) {
		GL_Graphic* new_graphic = nullptr;
		gl_display->surface() = true;
		switch (current_mode[id])
		{
			case 0: new_graphic = new GL_Graphic(gl_graphic->color()); gl_display->point() = true; gl_display->surface() = false;  gl_display->wireframe() = false; break;
			case 1: new_graphic = new GL_GraphicSurface(gl_graphic->color()); break;
			case 2: new_graphic = new GL_GraphicHighOrder(2, gl_graphic->color()); break;
			case 3: new_graphic = new GL_GraphicElement(gl_graphic->color()); break;
			default: break;
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

	if (current_mode[id] == 3) {
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




void UI_Data_Recorder::draw(Entity* entity) {
	int id = entity->id() - 1;
	if(saved.size() <= id) {
		saved.push_back(false);
		save_frame.push_back(0);
	}
	DataRecorder* data_recorder = entity->get_component<DataRecorder>();
	ImGui::Text(("File : " + data_recorder->json_path()).c_str());
	if (ImGui::Button("Save")) {
		data_recorder->save();
		saved[id] = true;
		save_frame[id] = Time::Frame();
	}
	if (saved[id]) {
		ImGui::SameLine();
		ImGui::Text("Last save at frame %d", save_frame[id]);
	}
}



void UI_Data_Displayer::draw(Entity* entity) {
	FEM_DataDisplay* data_recorder = entity->get_component<FEM_DataDisplay>();
	int _current_display = data_recorder->_mode;
	int _current_colormap = data_recorder->color_map;
	if (ImGui::Combo("Data Type", &_current_display, str_display_types.data(), static_cast<int>(str_display_types.size()))) {
		data_recorder->_mode = static_cast<FEM_DataDisplay::Type>(_current_display);
		//data_recorder->update();
	}
	if (ImGui::Combo("Color Map", &_current_colormap, str_colormap_types.data(), static_cast<int>(str_colormap_types.size()))) {
		data_recorder->color_map = static_cast<ColorMap::Type>(_current_colormap);
		//data_recorder->update();
	}
}

void UI_Graphic_Saver::draw(Entity* entity) {
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


void UI_PBD_Dynamic::draw(Entity* entity) {
	XPBD_FEM_Dynamic* pbd = entity->get_component<XPBD_FEM_Dynamic>();
	it = pbd->get_iteration();
	sub_it = pbd->get_sub_iteration();
	if (ImGui::InputInt("steps", &it) || ImGui::InputInt("sub steps", &sub_it)) {
		it = std::max(0, it);
		sub_it = std::max(0, sub_it);
		pbd->set_iterations(it, sub_it);
	}
}

void UI_Constraint_Rigid_Controller::draw(Entity* entity) {
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
