#pragma once
#include <vector>
#include <string>
#include "Pattern.h"
#include "Component.h"

struct Entity : public Behaviour {
    Entity(const int id);

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

    virtual ~Entity();

    int id() const { return _id; }
    std::string &name() { return _name; }

protected:
    int _id;
    std::string _name;
    std::vector<Behaviour *> _components;
};
