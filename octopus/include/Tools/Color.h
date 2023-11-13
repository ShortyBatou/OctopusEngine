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
    
