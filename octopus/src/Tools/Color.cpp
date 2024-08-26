#include "Tools/Color.h"

scalar ColorBase::Hue2RGB(scalar p, scalar q, scalar t) {
    if (t < 0.f) t += 1.f;
    if (t > 1.f) t -= 1.f;

    if (t < 1.f / 6.f) return p + (q - p) * 6.f * t;
    if (t < 1.f / 2.f) return q;
    if (t < 2.f / 3.f) return p + (q - p) * (2.f / 3.f - t) * 6;

    return p;
}

Color ColorBase::HSL2RGB(scalar h, scalar s, scalar l) {
    h /= 360.f;
    s /= 100.f;
    l /= 100.f;
    Color result;

    if (s == 0) {
        result.r = result.g = result.b = l;
        result.a = 1.f; // achromatic
    } else {
        float q = l < 0.5f ? l * (1.f + s) : l + s - l * s;
        float p = 2.f * l - q;
        result.r = Hue2RGB(p, q, h + 1.f / 3.f);
        result.g = Hue2RGB(p, q, h);
        result.b = Hue2RGB(p, q, h - 1.f / 3.f);
        result.a = 1.f;
    }

    return result;
}

std::string ColorMap::Type_To_Str(const Type type) {
    switch (type) {
        case Default: return "Default";
        case Rainbow: return "Rainbow";
        case Viridis: return "Viridis";
        case BnW: return "Black And White";
        default: return "None";
    }
}

Color ColorMap::evaluate(const scalar t) {
    const scalar n = static_cast<scalar>(_map[_type].size()) - 1;
    const int a = floor(t * n), b = ceil(t * n);
    const scalar x = t * static_cast<scalar>(n) - static_cast<scalar>(a);
    return glm::mix(_map[_type][a], _map[_type][b], x);
}

ColorMap::Type ColorMap::_type = Default;

std::map<ColorMap::Type, std::vector<Color> > ColorMap::_map = {
    {Default, {Color(0.2, 0.2, 0.9, 1.), ColorBase::White(), Color(0.9, 0.2, 0.2, 1.)}},
    {
        Rainbow,
        {
            Color(0.1f, 0.3f, 1.0f, 1.f), Color(0.1f, 0.85f, 0.4f, 1.f), Color(1.0f, 1.0f, 0.1f, 1.f),
            Color(1.0f, 0.5f, 0.3f, 1.f), Color(0.8f, 0.1f, 0.4f, 1.f)
        }
    },
    {
        Viridis,
        {
            Color(0.3f, 0.05f, 0.35f, 1.f), Color(0.25f, 0.45f, 0.7f, 1.f), Color(0.15f, 0.6f, 0.55f, 1.f),
            Color(0.5f, 0.8f, 0.3f, 1.f), Color(0.95f, 0.85f, 0.3f, 1.f)
        }
    },
    {BnW, {Color(0.), Color(1.)}}
};
