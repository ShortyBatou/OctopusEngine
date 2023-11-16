#pragma once
#include <string>
#include "Entity.h"
#include "Pattern.h"
class Engine : public Singleton<Engine>, public Behaviour {
    std::vector<Entity*> _entities; 
public:

    virtual void init() override
    { 
        for (unsigned int i = 0; i < _entities.size(); ++i)
            _entities[i]->init();  
    }

    virtual void late_init() override {
        for (unsigned int i = 0; i < _entities.size(); ++i)
            _entities[i]->late_init();
    }

    virtual void update() override
    {
        for (unsigned int i = 1; i < _entities.size(); ++i)
            if (_entities[i]->active()) _entities[i]->update();  
        
        _entities[0]->update(); // always update root at the end
    }

    virtual void late_update() override
    {
        for (unsigned int i = 1; i < _entities.size(); ++i)
            if (_entities[i]->active()) _entities[i]->late_update();

        _entities[0]->late_update();  // always update root at the end
    }

    static Entity* CreateEnity()
    {
        auto& engine = Engine::Instance();
        Entity* e = new Entity();
        engine._entities.push_back(e);
        return e;
    }

    static Entity* CreateEnity(const std::string& name)
    {
        auto& engine = Engine::Instance();
        Entity* e    = new Entity(name);
        engine._entities.push_back(e);
        return e;
    }


    static Entity* GetEntity(const std::string& name) {
        auto& engine = Engine::Instance();
        for(Entity* e : engine._entities)
            if(e->name() == name) return e;
        return nullptr;
    }
    
    static Entity* GetEntity(unsigned int id) {
        auto& engine = Engine::Instance();
        for (unsigned int i = 0; i < engine._entities.size(); ++i) 
            if (id == engine._entities[i]->id()) 
                return engine._entities[i];
        return nullptr;
    }

    static std::vector<Entity*>& GetEntities()
    { 
        auto& engine = Engine::Instance();
        return engine._entities;
    }

    static unsigned int Count()
    {
        auto& engine = Engine::Instance();
        return engine._entities.size();
    }

    void clear() {
        for(unsigned int i = 0; i < _entities.size(); ++i) {
            delete _entities[i];
        }
        _entities.clear();
    }

    virtual ~Engine() { clear(); }

};