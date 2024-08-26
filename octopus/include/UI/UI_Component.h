#pragma once
#include <Script/VTK/VTK_FEM.h>

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
class UI_SceneManager final : public UI_Display {
public:
	UI_SceneManager() : _item_current(SceneManager::SceneID()) 
	{ 
		std::cout << "UI SCENE " << _item_current << std::endl;
	}

	[[nodiscard]] virtual std::string name() const override {
		return "Scene";
	}

	void draw() override;
protected:
	int _item_current;
};

class UI_DisplaySettings final : public UI_Component {
public:
	[[nodiscard]] std::string name() const override { return "Colors";}

	bool can_draw(Entity* entity) override {
		return entity->get_component<OpenGLManager>() != nullptr;
	}

	void draw(Entity* entity) override;
};

class UI_Time final : public UI_Component {
public:
	UI_Time() : UI_Component(), _fixed_delta_t(0) {}
	[[nodiscard]] std::string name() const override {
		return "Time";
	}
	void init() override {
		_fixed_delta_t = Time::Fixed_DeltaTime();
	}

	bool can_draw(Entity* entity) override {
		return true;
	}

	void draw(Entity* entity) override;

protected:
	scalar _fixed_delta_t;
};

class UI_Dynamic final : public UI_Component {
public:
	UI_Dynamic() : UI_Component(), _gravity({}) {}

	[[nodiscard]] std::string name() const override { return "Dynamic"; }
	void init() override { _gravity = Dynamic::gravity(); }

	bool can_draw(Entity* entity) override { return true; }

	void draw(Entity* entity) override;
protected:
	Vector3 _gravity;
};


class UI_Camera final : public UI_Component {
public:
	[[nodiscard]] std::string name() const override { return "Camera"; }

	bool can_draw(Entity* entity) override {
		return entity->get_component<CameraManager>() != nullptr;
	}

	 void draw(Entity* entity) override;
};



class UI_Mesh_Display final : public UI_Component {
public:
	UI_Mesh_Display() : UI_Component(), current_mode(0) {}

	[[nodiscard]] std::string name() const override { return "Mesh Display"; }

	bool can_draw(Entity* entity) override {
		return entity->get_component<GL_DisplayMesh>() != nullptr;
	}
	int get_mode(Entity* ) {

	}
	void draw(Entity* entity) override;

private:
	std::vector<int> current_mode;
};


class UI_Data_Recorder final : public UI_Component {
public:
	UI_Data_Recorder() : UI_Component(), saved({}), save_frame({}){
	}

	[[nodiscard]] std::string name() const override { return "Data Recorder"; }


	bool can_draw(Entity* entity) override {
		return entity->get_component<DataRecorder>() != nullptr;
	}

	 void draw(Entity* entity) override;

protected: 
	std::vector<bool> saved;
	std::vector<int> save_frame;
};

class UI_Data_Displayer final : public UI_Component {
public:
	UI_Data_Displayer() : UI_Component() {
		for (int d = FEM_DataDisplay::Type::BaseColor; d != FEM_DataDisplay::Type::None; ++d) {
			std::string str = FEM_DataDisplay::Type_To_Str(static_cast<FEM_DataDisplay::Type>(d));
			const char* c_str = str.c_str();
			char* copy_str = new char[strlen(c_str) + 1]; strcpy(copy_str, c_str);
			std::cout << c_str << " " << copy_str << " " << FEM_DataDisplay::Type_To_Str(static_cast<FEM_DataDisplay::Type>(d)) << std::endl;
			str_display_types.push_back(copy_str);
		}
		for (int d = ColorMap::Type::Default; d != ColorMap::Type::BnW + 1; ++d) {
			std::string str = ColorMap::Type_To_Str(static_cast<ColorMap::Type>(d));
			const char* c_str = str.c_str();
			char* copy_str = new char[strlen(c_str) + 1]; strcpy(copy_str, c_str);
			str_colormap_types.push_back(copy_str);
		}
	}

	[[nodiscard]] std::string name() const override { return "FEM Data Displayer";}

	bool can_draw(Entity* entity) override {
		return entity->get_component<FEM_DataDisplay>() != nullptr;
	}

	void draw(Entity* entity) override;
	~UI_Data_Displayer() override {
		for(auto str : str_colormap_types) {delete str;}
		for(auto str : str_display_types) {delete str;}
	}
protected:
	
	std::vector<char*> str_colormap_types;
	std::vector<char*> str_display_types;
};

class UI_Graphic_Saver final : public UI_Component {
public:
	UI_Graphic_Saver() : UI_Component(), save_frame(0), saved(false) {}

	[[nodiscard]] std::string name() const override { return "Graphic Mesh Saver"; }


	bool can_draw(Entity* entity) override {
		return entity->get_component<VTK_Graphic>() != nullptr;
	}

	void draw(Entity* entity) override;
protected:
	bool saved;
	int save_frame;
};


class UI_PBD_Dynamic final : public UI_Component {
public:
	UI_PBD_Dynamic() : UI_Component(), it(0), sub_it(0){
	}

	[[nodiscard]] std::string name() const override { return "PBD"; }


	bool can_draw(Entity* entity) override {
		return entity->get_component<XPBD_FEM_Dynamic>() != nullptr;
	}

	void draw(Entity* entity) override;

protected:
	int it, sub_it;
};


class UI_Constraint_Rigid_Controller final : public UI_Component {
public:
	UI_Constraint_Rigid_Controller() : UI_Component() { }

	[[nodiscard]] std::string name() const override { return "Constraint Rigid"; }


	bool can_draw(Entity* entity) override {
		return entity->get_component<Constraint_Rigid_Controller>() != nullptr;
	}

	void draw(Entity* entity) override;
};