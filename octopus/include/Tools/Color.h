#pragma once
#include "Core/Base.h"
#include <string>
#include <map>
#include <vector>

struct ColorBase {
    static Vector4 White() { return {1.f, 1.f, 1.f, 1.f}; }
    static Vector4 Red() { return {1.f, 0.f, 0.f, 1.f}; }
    static Vector4 Green() { return {0.f, 1.f, 0.f, 1.f}; }
    static Vector4 Blue() { return {0.f, 0.f, 1.f, 1.f}; }
    static Vector4 Yellow() { return {1.f, 1.f, 0.f, 1.f}; }
    static Vector4 Magenta() { return {1.f, 0.f, 1.f, 1.f}; }
    static Vector4 Cyan() { return {0.f, 1.f, 1.f, 1.f}; }
    static Vector4 Grey(scalar s = 0.5f) { return {s, s, s, 1.f}; }
    static Vector4 Black() { return {0.f, 0.f, 0.f, 1.f}; }

    static float Hue2RGB(scalar p, scalar q, scalar t);

    static Color HSL2RGB(scalar h, scalar s, scalar l);
};

struct ColorMap {
    enum Type { Default, Rainbow, Viridis, BnW };

    static std::string Type_To_Str(Type type);

    static Color evaluate(scalar t);

    static Type &Get_Type() { return _type; }
    static void Set_Type(const Type type) { _type = type; }

private:
    static Type _type;
    static std::map<Type, std::vector<Color> > _map;
};
