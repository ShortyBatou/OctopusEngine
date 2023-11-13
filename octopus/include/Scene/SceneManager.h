#pragma once
#include <iostream>
#include "Core/Engine.h"
#include "Scene.h"
#include "BaseScene.h"
class SceneManager
{
public:
    SceneManager() {
        _scenes.push_back(new BaseScene);
    }

    void build(unsigned int scene_id = 0) { 
        std::cout << "Build Scene : " << scene_id << " " << _scenes[scene_id]->name() << std::endl;
        std::cout << "Clear Entities" << std::endl;
        Engine::Instance().clear();
        Entity* root = Engine::CreateEnity("root");

        std::cout << "Init before build" << std::endl;
        _scenes[scene_id]->init();

        std::cout << "Build root" << std::endl;
        _scenes[scene_id]->build_root(root);

        std::cout << "Build entities" << std::endl;
        _scenes[scene_id]->build_entities();

        std::cout << "Init Scene" << std::endl;
        Engine::Instance().init();
    }

    virtual ~SceneManager()
    {
        for (Scene* s : _scenes) delete s;
        _scenes.clear();
    }

protected:
    std::vector<Scene*> _scenes;
};