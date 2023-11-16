#pragma once
#include <vector>
#include <string>
#include "Pattern.h"
#include "Component.h"

struct Entity : public Behaviour, public ID_Creator<Entity> {
    Entity() {
        _name = "Entity_" + std::to_string(this->id());
    }
    Entity(std::string name) : _name(name) {}

    virtual void init() override
    {
        for (unsigned int i = 0; i < _components.size(); ++i)
            _components[i]->init();

    }

    virtual void late_init() override {
        for (unsigned int i = 0; i < _components.size(); ++i)
            _components[i]->late_init();
    }

    virtual void update() override
    {
        for (unsigned int i = 0; i < _components.size(); ++i)
            if (_components[i]->active()) _components[i]->update();     
    }

    virtual void late_update() override
    {
        for (unsigned int i = 0; i < _components.size(); ++i)
            if (_components[i]->active()) _components[i]->late_update();
    }

    template<class T>
    T* getComponent() {
        for (unsigned int i = 0; i < _components.size(); ++i)
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
        for (unsigned int i = 0; i < _components.size(); ++i)
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
    T* getComponentById(unsigned int i) {
        if(i >= _components.size()) return nullptr;
        
        return return std::static_cast<T>(_components[i]);
    }

    void addBehaviour(Behaviour* behaviour) {
        _components.push_back(behaviour);
    }

    void addComponent(Component* component) {
        component->_entity = this;
        _components.push_back(component);
    }

    virtual ~Entity(){
        for(unsigned int i = 0; i < _components.size(); ++i){
            delete _components[i];
        }
        _components.clear();
    }

    std::string& name() {return _name;}
protected: 
    std::string _name;
    std::vector<Behaviour*> _components;
};