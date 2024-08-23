#pragma once
#include "Core/Base.h"
#include <random>

struct Random {
    static scalar Eval() { return static_cast<scalar>(rand())/RAND_MAX;}

    template<typename T>
    static T Range(const T& min, const T& max) { return range(Eval(), min, max); }
};

