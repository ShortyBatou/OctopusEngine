#pragma once
#include "Core/Base.h"
#include <vector>
enum Material {
    Hooke, StVK, Neo_Hooke, Developed_Neohooke
};

std::string get_material_name(Material material) {
    switch (material)
    {
        case Hooke: return "Hooke";
        case StVK: return "SaintVenant";
        case Neo_Hooke: return "Stable Neo-Hooke";
        case Developed_Neohooke: return "Stable Neo-Hooke";
        default: return "";
    }
}

struct ContinuousMaterial {
    scalar lambda, mu;
    scalar young, poisson;
    

    ContinuousMaterial(const scalar _young, const scalar _poisson) : young(_young), poisson(_poisson) {
        lambda = computeLambda();
        mu = computeMu();
    }
    scalar computeLambda() { return young * poisson / ((1. + poisson) * (1. - 2. * poisson)); }
    scalar computeMu() { return young / (2. * (1. + poisson)); }

    static Matrix3x3 getStrainTensorLinear(const Matrix3x3& F)
    {
        return scalar(0.5) * (glm::transpose(F) + F) - Matrix::Identity3x3();
    }

    static Matrix3x3 getStrainTensor(const Matrix3x3& F)
    {
        return scalar(0.5) * (glm::transpose(F) * F - Matrix::Identity3x3());
    }
    virtual ~ContinuousMaterial() {}
};

