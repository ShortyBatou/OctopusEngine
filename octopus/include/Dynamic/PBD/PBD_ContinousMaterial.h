#pragma once
#include "Dynamic/FEM/ContinousMaterial.h"
struct PBD_ContinuousMaterial : public ContinuousMaterial {
    PBD_ContinuousMaterial(const scalar _young, const scalar _poisson) : ContinuousMaterial(_young, _poisson) { }
    virtual void getStressTensor(const Matrix3x3& F, Matrix3x3& P) {}
    virtual scalar getEnergy(const Matrix3x3& F) { return 0.; }
    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) = 0;
    virtual scalar getStiffness() const { return young; }
    virtual ~PBD_ContinuousMaterial() {}
};

struct Hooke_First : public PBD_ContinuousMaterial {
    Hooke_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override {
        const auto trace = Matrix::Trace(getStrainTensorLinear(F));
        // P(F) = 2E
        // C(F) = tr(E)�
        energy = trace * trace;
        P = scalar(2.) * trace * Matrix::Identity3x3();
    }

    virtual scalar getStiffness() const override { return this->lambda; }
};

struct Hooke_Second : public PBD_ContinuousMaterial {

    Hooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override {
        const auto E = getStrainTensorLinear(F);
        // P(F) = 2E
        // C(F) = tr(E�)
        P = scalar(2.) * E;
        energy = Matrix::SquaredTrace(E);
    }

    virtual scalar getStiffness() const override { return this->mu * 2.; }
};

struct StVK_First : public PBD_ContinuousMaterial {
    StVK_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override {
        const auto trace = Matrix::Trace(getStrainTensor(F));
        // P(F) = 2 tr(E) F 
        P = scalar(2) * trace * F;

        // C(F) = tr(E)^2
        energy = trace * trace;
    }
    virtual scalar getStiffness() const override { return this->lambda; }
};

struct StVK_Second : public PBD_ContinuousMaterial {

    StVK_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override {
        const auto E = getStrainTensor(F);
        // P(F) = 2 F E
        P =  scalar(4) * F * E;
        // C(F) = tr(E^2)
        energy = scalar(2) * Matrix::SquaredTrace(E);
    }
    virtual scalar getStiffness() const override { return this->mu; }
};

struct VolumePreservation : public PBD_ContinuousMaterial {

    scalar alpha;

    VolumePreservation(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson), alpha(1) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        // C(F) = (det(F) - alpha)�
        // P(F) = 2 det(F) det(F)/dx
        scalar I_3 = glm::determinant(F);
        scalar detF = I_3 - alpha;
        scalar shift = this->mu / this->lambda;
        energy = (detF) * (detF);
        Matrix3x3 d_detF; // derivative of det(F) by F
        d_detF[0] = glm::cross(F[1], F[2]);
        d_detF[1] = glm::cross(F[2], F[0]);
        d_detF[2] = glm::cross(F[0], F[1]);
        P = scalar(2) * detF * d_detF;
    }

    virtual scalar getStiffness() const override { return this->lambda; }
};




struct Stable_NeoHooke_First : public VolumePreservation {
    Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : VolumePreservation(_young, _poisson) { 
        this->alpha = scalar(1) - this->mu / this->lambda;
    }
};

struct Stable_NeoHooke_Second : public PBD_ContinuousMaterial {
    Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        energy = Matrix::SquaredNorm(F) - scalar(3);

        P = scalar(2) * F;
    }

    virtual scalar getStiffness() const override { return this->mu; }
};


struct Developed_Stable_NeoHooke_First : public VolumePreservation {
    Developed_Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : VolumePreservation(_young, _poisson) { }
};

struct Developed_Stable_NeoHooke_Second : public PBD_ContinuousMaterial {
    Developed_Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        scalar I_3 = glm::determinant(F);
        energy = Matrix::SquaredNorm(F) - scalar(3) - scalar(2) * (I_3 - scalar(1));

        Matrix3x3 d_detF; // derivative of det(F) by F
        d_detF[0] = glm::cross(F[1], F[2]);
        d_detF[1] = glm::cross(F[2], F[0]);
        d_detF[2] = glm::cross(F[0], F[1]);
        P = scalar(2) * F - scalar(2) * d_detF;
    }

    virtual scalar getStiffness() const override { return this->mu; }
};


struct Anysotropic : public PBD_ContinuousMaterial {
    Vector3 a;
    Anysotropic(const scalar _young, const scalar _poisson, const Vector3& _a) : PBD_ContinuousMaterial(_young, _poisson), a(_a) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        Vector3 Fa = F * a;
        scalar IVc_1 = glm::dot(Fa, Fa) - scalar(1);

        energy = (IVc_1) * (IVc_1);

        P = scalar(4) * IVc_1 * glm::outerProduct(Fa, a);
    }

    virtual scalar getStiffness() const override { return this->mu; }
};


struct Sqrt_Anysotropic : public PBD_ContinuousMaterial {
    Vector3 a;
    Sqrt_Anysotropic(const scalar _young, const scalar _poisson, const Vector3& _a) : PBD_ContinuousMaterial(_young, _poisson), a(_a) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        Vector3 Fa = F * a;
        scalar dFa = glm::length(Fa);

        energy = (dFa - scalar(1)) * (dFa - scalar(1));

        P = scalar(2) * (scalar(1) - scalar(1) / dFa) * glm::outerProduct(Fa, a);
    }

    virtual scalar getStiffness() const override { return this->mu; }
};

// Combine multiple energies that have the same stifness factor
struct Material_Union : public PBD_ContinuousMaterial {
    std::vector<PBD_ContinuousMaterial*> materials;

    Material_Union(std::vector<PBD_ContinuousMaterial*> _materials) : PBD_ContinuousMaterial(_materials[0]->young, _materials[0]->poisson), materials(_materials) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        Matrix3x3 temp_P;
        scalar temp_E;
        energy = 0;
        P = Matrix::Zero3x3();
        for (PBD_ContinuousMaterial* m : materials) {
            m->getStressTensorAndEnergy(F, temp_P, temp_E);
            energy += temp_E;
            P += temp_P;
        }
    }

    virtual scalar getStiffness() const override { return materials[0]->getStiffness(); }
    virtual ~Material_Union() {
        for (PBD_ContinuousMaterial* m : materials) delete m;
        materials.clear();
    }

};


//Muller and Macklin neohooke energy for Tetra
struct C_Stable_NeoHooke_First : public PBD_ContinuousMaterial {
    scalar alpha;

    C_Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) {
        this->alpha = 1 + this->mu / this->lambda;
    }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        // C(F) = (det(F) - alpha)
        // P(F) = 2 det(F) det(F)/dx
        energy = glm::determinant(F) - alpha;
        P[0] = glm::cross(F[1], F[2]);
        P[1] = glm::cross(F[2], F[0]);
        P[2] = glm::cross(F[0], F[1]);
    }

    virtual scalar getStiffness() const override { return this->lambda; }
};


struct C_Stable_NeoHooke_Second : public PBD_ContinuousMaterial {
    C_Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        energy = sqrt(abs(Matrix::SquaredNorm(F) - 3));
        P = F * (scalar(1) / energy);
    }

    virtual scalar getStiffness() const override { return this->mu; }
};


struct C_Developed_Stable_NeoHooke_First : public PBD_ContinuousMaterial {
    C_Developed_Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        // C(F) = (det(F) - 1)
        // P(F) = det(F)/dx
        energy = glm::determinant(F) - 1;
        P[0] = glm::cross(F[1], F[2]);
        P[1] = glm::cross(F[2], F[0]);
        P[2] = glm::cross(F[0], F[1]);
    }

    virtual scalar getStiffness() const override { return this->lambda; }
};


struct C_Developed_Stable_NeoHooke_Second : public PBD_ContinuousMaterial {
    C_Developed_Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        energy = Matrix::SquaredNorm(F) - 3 - scalar(2) * (glm::determinant(F) - 1);
        energy = sqrt(abs(energy));

        Matrix3x3 d_det;
        d_det[0] = glm::cross(F[1], F[2]);
        d_det[1] = glm::cross(F[2], F[0]);
        d_det[2] = glm::cross(F[0], F[1]);
        P = (scalar(2) * F - scalar(2) * d_det) * (scalar(1) / (scalar(2)*energy));
    }

    virtual scalar getStiffness() const override { return this->mu; }
};