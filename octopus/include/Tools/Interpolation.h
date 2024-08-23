#pragma once
#include "Core/Base.h"

template<typename T>
T linear_interpolation(scalar t, const T& min, const T& max) {
    return t * (max - min) + min;
}

template<typename T>
T range(float t, const T& min, const T& max) {
    return linear_interpolation(t, min, max);
}