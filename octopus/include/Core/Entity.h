#pragma once
#include <vector>
#include <string>
#include "Pattern.h"
#include "Component.h"

struct Entity : public Behaviour {
    Entity(int id) : _id(id) {
        _name = "Entity_" + std::to_string(id);
    }
    Entity(std::string name, int id) : _name(name), _id(id) {}

    virtual void init() override
    {
        for (int i = 0; i < _components.size(); ++i)
            _components[i]->init();

    }

    virtual void late_init() override {
        for (int i = 0; i < _components.size(); ++i)
            _components[i]->late_init();
    }

    virtual void update() override
    {
        for (int i = 0; i < _components.size(); ++i)
            if (_components[i]->active()) _components[i]->update();     
    }

    virtual void late_update() override
    {
        for (int i = 0; i < _components.size(); ++i)
            if (_components[i]->active()) _components[i]->late_update();
    }

    template<class T>
    T* getComponent() {
        for (int i = 0; i < _components.size(); ++i)
        {
            if (typeid(*_components[i]) == typeid(T))
                return dynamic_cast<T*>(_components[i]);
            
            T* _c = dynamic_cast<T*>(_components[i]);
            if (_c != nullptr) return _c;
        }
            
        return nullptr;
    }

    template<class T>
    std::vector<T*> getComponents() {
        std::vector<T*> compontents;
        for (int i = 0; i < _components.size(); ++i)
        {
            if (typeid(*_components[i]) == typeid(T)) {
                compontents.push_back(static_cast<T*>(_components[i]));
            }
            else {
                T* _c = dynamic_cast<T*>(_components[i]);
                if (_c != nullptr)
                    compontents.push_back(_c);
            }
                

            
            
        }
        return compontents;
    }

    template<class T>
    T* getComponentById(int i) {
        if(i >= _components.size()) return nullptr;
        
        return static_cast<T>(_components[i]);
    }

    void addBehaviour(Behaviour* behaviour) {
        _components.push_back(behaviour);
    }

    void addComponent(Component* component) {
        component->_entity = this;
        _components.push_back(component);
    }

    virtual ~Entity(){
        for(int i = 0; i < _components.size(); ++i){
            delete _components[i];
        }
        _components.clear();
    }
    inline int id() { return _id; }
    std::string& name() {return _name;}
protected: 
    int _id;
    std::string _name;
    std::vector<Behaviour*> _components;
};
