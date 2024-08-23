#pragma once
#include <string>
#include "Entity.h"
#include "Pattern.h"

class Engine : public Singleton<Engine>, public Behaviour {
    std::vector<Entity*> _entities;

public:
    void init() override;
    void late_init() override;
    void update() override;
    void late_update() override;

    static Entity* CreateEnity();
    static Entity* CreateEnity(const std::string& name);
    static Entity* GetEntity(const std::string& name);
    static Entity* GetEntity(int id);
    static std::vector<Entity*>& GetEntities();
    static int Count();

    void clear();
    ~Engine() override;
};