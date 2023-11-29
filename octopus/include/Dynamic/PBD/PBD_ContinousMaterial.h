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
        // C(F) = tr(E)²
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
        // C(F) = tr(E²)
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
        P = scalar(2.) * trace * F;

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
        P = scalar(2.) * F * E;
        // C(F) = tr(E^2)
        energy = Matrix::SquaredTrace(E);
    }
    virtual scalar getStiffness() const override { return this->mu * 2.; }
};

struct VolumePreservation : public PBD_ContinuousMaterial {

    scalar alpha;

    VolumePreservation(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson), alpha(1) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        // C(F) = (det(F) - alpha)²
        // P(F) = 2 det(F) det(F)/dx
        scalar I_3 = glm::determinant(F);
        scalar detF = I_3 - alpha;
        energy = (detF) * (detF) ;
        Matrix3x3 d_detF; // derivative of det(F) by F
        d_detF[0] = glm::cross(F[1], F[2]);
        d_detF[1] = glm::cross(F[2], F[0]);
        d_detF[2] = glm::cross(F[0], F[1]);
        P = scalar(2.) * detF * d_detF;
    }

    virtual scalar getStiffness() const override { return this->lambda; }
};


struct Stable_NeoHooke_First : public VolumePreservation {
    Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : VolumePreservation(_young, _poisson) {
        this->alpha = 1 + this->mu / this->lambda;
    }
};


struct Stable_NeoHooke_Second : public PBD_ContinuousMaterial {
    Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        energy = Matrix::SquaredNorm(F) - 3.;
        P = scalar(2.) * F;
    }

    virtual scalar getStiffness() const override { return this->mu; }
};

