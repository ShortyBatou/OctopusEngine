#pragma once
#include "Dynamic/FEM/ContinuousMaterial.h"
/*
struct SVD_ContinuousMaterial : public ContinuousMaterial {
    SVD_ContinuousMaterial(const scalar _young, const scalar _poisson) : ContinuousMaterial(_young, _poisson) { }
    virtual scalar get_energy(const Vector3& s) = 0;
    virtual void get_constraint(const Vector3& s, Vector3& C) = 0;
    virtual void get_stiffness(const Vector3& s, Matrix3x3& A) = 0;
    virtual scalar get_energy(const Matrix3x3& F) {
        return 0;
    }
    virtual Matrix3x3 get_pk1(const Matrix3x3& F) {
        return Matrix::Identity3x3();
    }
    virtual void get_constraint_and_stiffness(const Vector3& s, Vector3& J, scalar& energy) {};
    virtual ~SVD_ContinuousMaterial() {}
};

struct SVD_StVK : public SVD_ContinuousMaterial {
    SVD_StVK(const scalar _young, const scalar _poisson) : SVD_ContinuousMaterial(_young, _poisson) { }
    virtual scalar get_energy(const Vector3& s) {
        scalar s0 = s.x * s.x;
        scalar s1 = s.y * s.y;
        scalar s2 = s.z * s.z;
        scalar I_c = s0 + s1 + s2;
        scalar II_c = s0 * s0 + s1 * s1 + s2 * s2;
        return this->lambda / scalar(8) * (I_c - 3) * (I_c - 3) + this->mu / 2 * (II_c - 2 * I_c + 3);
    }

    virtual void get_constraint(const Vector3& s, Vector3& C) {
        scalar I_c = s.x * s.x + s.y * s.y + s.z * s.z - 3;
        C.x = s.x * (lambda / 2 * I_c + mu * (s.x * s.x - 1));
        C.y = s.y * (lambda / 2 * I_c + mu * (s.y * s.y - 1));
        C.z = s.z * (lambda / 2 * I_c + mu * (s.z * s.z - 1));
    }

    virtual void get_stiffness(const Vector3& s, Matrix3x3& K) {
        scalar s0 = s.x * s.x;
        scalar s1 = s.y * s.y;
        scalar s2 = s.z * s.z;

        scalar I_c = s0 + s1 + s2 - 3;
        K[0][0] = lambda * (I_c + s0 * 2) + mu * (3 * s0 - 1);
        K[1][1] = lambda * (I_c + s1 * 2) + mu * (3 * s1 - 1);
        K[2][2] = lambda * (I_c + s2 * 2) + mu * (3 * s2 - 1);


        K[1][0] = lambda * s.x * s.y * 2;
        K[2][0] = lambda * s.x * s.z * 2; 
        K[1][2] = lambda * s.y * s.z * 2;

        K[0][1] = K[1][0];
        K[0][2] = K[2][0];
        K[2][1] = K[1][2];
    }
};


struct SVD_Stable_Neohooke : public SVD_ContinuousMaterial {
    scalar alpha;
    SVD_Stable_Neohooke(const scalar _young, const scalar _poisson) : SVD_ContinuousMaterial(_young, _poisson) { 
        alpha = scalar(1) - mu / lambda;
    }
    virtual scalar get_energy(const Vector3& s) {
        scalar D = s.x * s.y * s.z - alpha;
        return this->mu / scalar(2) * (s.x * s.x + s.y * s.y + s.z * s.z) + this->lambda / 2 * (D * D);
    }

    virtual void get_constraint(const Vector3& s, Vector3& C) {
        scalar D = s.x * s.y * s.z - 1;
        C.x = this->mu * (s.x - s.y * s.z) + this->lambda * D * s.y * s.z;
        C.y = this->mu * (s.y - s.x * s.z) + this->lambda * D * s.x * s.z;
        C.z = this->mu * (s.z - s.x * s.y) + this->lambda * D * s.x * s.y;
    }

    virtual void get_stiffness(const Vector3& s, Matrix3x3& K) {
        scalar k0 = 2 * (s.x * s.y * s.z - this->alpha);

        K[0][0] = lambda * (s.y * s.z) * (s.y * s.z) + mu;
        K[1][1] = lambda * (s.x * s.z) * (s.x * s.z) + mu;
        K[2][2] = lambda * (s.x * s.y) * (s.x * s.y) + mu;
        K[1][0] = lambda * k0 * s.z;
        K[2][0] = lambda * k0 * s.y;
        K[1][2] = lambda * k0 * s.x;

        K[0][1] = K[1][0];
        K[0][2] = K[2][0];
        K[2][1] = K[1][2];
    }
};

SVD_ContinuousMaterial* get_svd_materials(Material material, scalar young, scalar poisson) {
    SVD_ContinuousMaterial* m = nullptr;
    switch (material)
    {
    case StVK:
        m = new SVD_StVK(young, poisson);
        break;
    case Neo_Hooke:
        m = new SVD_Stable_Neohooke(young, poisson);
        break;
    case Developed_Neohooke:
        m = new SVD_Stable_Neohooke(young, poisson);
        break;
    default:
        std::cout << "Material not found" << std::endl;
        break;
    }
    return m;
}*/