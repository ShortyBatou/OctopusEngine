#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
struct Dynamic : Singleton<Dynamic> {
    friend Singleton<Dynamic>;

    Dynamic(const Vector3& g = Vector3(0, -9.81, 0), const float& damp = 0.f)
        : _gravity(g) {}

    void init() { }

    void update() { }

    virtual ~Dynamic() {}

    static Vector3 gravity() { return Instance()._gravity; }
    void set_gravity(const Vector3& g) { _gravity = g; }
private:
    Vector3 _gravity;
};

struct DynamicManager : Behaviour {
    DynamicManager(const Vector3& g = Vector3(0, -9.81, 0)) {
        Dynamic* dynamic = Dynamic::Instance_ptr();
        dynamic->set_gravity(g);
    }

    ~DynamicManager() {
        Dynamic::Delete();
    }

};