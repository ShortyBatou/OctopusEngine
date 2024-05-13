#pragma once
#include "Core/Base.h"

struct ColorBase
{
    static Vector4 White() { return Vector4(1., 1., 1., 1.); }
    static Vector4 Red() { return Vector4(1., 0., 0., 1.); }
    static Vector4 Green() { return Vector4(0., 1., 0., 1.); }
    static Vector4 Blue() { return Vector4(0., 0., 1., 1.); }
    static Vector4 Yellow() { return Vector4(1., 1., 0., 1.); }
    static Vector4 Magenta() { return Vector4(1., 0., 1., 1.); }
    static Vector4 Cyan() { return Vector4(0., 1., 1., 1.); }
    static Vector4 Grey(scalar s = 0.5) { return Vector4(s, s, s, 1.); }
    static Vector4 Black() { return Vector4(0., 0., 0., 1.); }
};
    


struct ColorMap {
    enum Type { Default, Rainbow, Viridis };
    
    static Color evaluate(scalar t) {
        int n = _map[_type].size() - 1;
        int a = floor(t * n), b = ceil(t * n);
        scalar x = t * n - a;
        return glm::mix(_map[_type][a], _map[_type][b], x);
    }
   
    static ColorMap::Type& Get_Type() { return _type; }
    static void Set_Type(ColorMap::Type type) { _type = type; }
    
private:
    static ColorMap::Type _type;
    static std::map< ColorMap::Type, std::vector<Color> > _map;
};

ColorMap::Type ColorMap::_type = ColorMap::Type::Default;

std::map< ColorMap::Type, std::vector<Color> > ColorMap::_map = {
    {ColorMap::Type::Default, {Color(0.2, 0.2, 0.9, 1.), ColorBase::White(), Color(0.9, 0.2, 0.2, 1.)}},
    {ColorMap::Type::Rainbow, {Color(0.1f, 0.3f, 1.0f, 1.f), Color(0.1f, 0.85f, 0.4f, 1.f), Color(1.0f, 1.0f, 0.1f, 1.f), Color(1.0f, 0.5f, 0.3f, 1.f), Color(0.8f, 0.1f, 0.4f, 1.f)}},
    {ColorMap::Type::Viridis, {Color(0.3f , 0.05f, 0.35f, 1.f), Color(0.25f, 0.45f, 0.7f , 1.f), Color(0.15f, 0.6f , 0.55f, 1.f), Color(0.5f , 0.8f , 0.3f , 1.f), Color(0.95f, 0.85f, 0.3f , 1.f)}}
};


