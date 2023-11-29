#pragma once
#include <iostream>
#include "Core/Engine.h"
#include "Scene.h"
#include "UI/UI_Editor.h"
#include "Core/Pattern.h"

class SceneManager : public Singleton<SceneManager>
{
public:
    friend Singleton<SceneManager>;
    SceneManager() {
        _scene_id = 0;
    }

    void load_scene(UI_Editor* editor) {
        _need_to_load = false;

        std::cout << "Build Scene : " << _scene_id << " " << _scenes[_scene_id]->name() << std::endl;
        std::cout << "Clear Entities" << std::endl;
        Engine::Instance().clear();
        Entity* root = Engine::CreateEnity("root");

        std::cout << "Init before build" << std::endl;
        _scenes[_scene_id]->init();

        std::cout << "Build root" << std::endl;
        _scenes[_scene_id]->build_root(root);

        std::cout << "Build entities" << std::endl;
        _scenes[_scene_id]->build_entities();

        std::cout << "Build UI" << std::endl;
        _scenes[_scene_id]->build_editor(editor);

        std::cout << "Init Scene" << std::endl;
        Engine::Instance().init();

        std::cout << "Late Init Scene" << std::endl;
        Engine::Instance().late_init();
    }
    
    static void SetScene(unsigned int i) {
        if (i >= Instance()._scenes.size()) {
            std::cout << "The scene number " << i << " does not exist" << std::endl;
            return;
        }
        Instance()._need_to_load = true;
        Instance()._scene_id = i;
    }

    static std::vector<Scene*>& Scenes() { return Instance()._scenes; }
    static unsigned int SceneID() { return Instance()._scene_id; }
    static void Add(Scene* scene) { Instance()._scenes.push_back(scene); }

    virtual ~SceneManager()
    {
        for (Scene* s : _scenes) delete s;
        _scenes.clear();
    }

    bool need_to_load() {
        return _need_to_load;
    }

protected:
    bool _need_to_load;
    std::vector<Scene*> _scenes;
    unsigned int _scene_id;
};