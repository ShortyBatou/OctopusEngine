#pragma once
#include <vector>
#include <string>
#include "Pattern.h"
#include "Component.h"

struct Entity final : Behaviour {
    explicit Entity(int id);

    Entity(const std::string &name, int id);

    void init() override;

    void late_init() override;

    void update() override;

    void late_update() override;

    template<class T>
    T *get_component();

    template<class T>
    std::vector<T *> get_components();

    template<class T>
    T *get_component_by_id(int i);

    void add_behaviour(Behaviour *behaviour);

    void add_component(Component *component);

    void remove_component(Component *component);

    ~Entity() override;

    [[nodiscard]] int id() const { return _id; }
    std::string &name() { return _name; }

protected:
    int _id;
    std::string _name;
    std::vector<Behaviour *> _components;
};

template<class T>
T *Entity::get_component() {
    for (auto &_component: _components) {
        if (typeid(*_component) == typeid(T))
            return dynamic_cast<T *>(_component);

        T *_c = dynamic_cast<T *>(_component);
        if (_c != nullptr) return _c;
    }

    return nullptr;
}

template<class T>
std::vector<T *> Entity::get_components() {
    std::vector<T *> list;
    for (auto &_component: _components) {
        if (typeid(*_component) == typeid(T)) {
            list.push_back(static_cast<T *>(_component));
        } else {
            T *_c = dynamic_cast<T *>(_component);
            if (_c != nullptr)
                list.push_back(_c);
        }
    }
    return list;
}

template<class T>
T *Entity::get_component_by_id(int i) {
    if (i >= _components.size()) return nullptr;

    return static_cast<T>(_components[i]);
}