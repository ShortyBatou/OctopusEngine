#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"

struct Dynamic : Singleton<Dynamic> {
    friend Singleton;

    explicit Dynamic(const Vector3 &g = Vector3(0, -9.81, 0), const float &damp = 0.f)
        : _gravity(g) {
    }

    static Vector3 gravity() { return Instance()._gravity; }
    void set_gravity(const Vector3 &g) { _gravity = g; }

private:
    Vector3 _gravity;
};

struct DynamicManager final : Behaviour {
    explicit DynamicManager(const Vector3 &g = Vector3(0, -9.81, 0)) {
        Dynamic::Instance_ptr()->set_gravity(g);
    }

    ~DynamicManager() override {
        Dynamic::Delete();
    }
};
