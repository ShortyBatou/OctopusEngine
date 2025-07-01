#pragma once
#include "Core/Base.h"
#include "Tools/Interpolation.h"
#include <random>

struct Random {
    static scalar Eval() { return static_cast<scalar>(rand())/RAND_MAX;}

    static Vector2 UnitCircle() {
        const scalar rho = Eval() * PI * 2.f;
        return Vector2(cos(rho), sin(rho)) * Eval();
    }

    static Vector3 InBox(const Vector3& p_min, const Vector3& p_max) {
        return p_min + Vector3(p_max.x * Eval(), p_max.y * Eval(), p_max.z * Eval());
    }

    template<typename T>
    static T Range(const T& min, const T& max) { return range(Eval(), min, max); }
};

