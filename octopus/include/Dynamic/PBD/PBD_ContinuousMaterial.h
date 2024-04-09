#pragma once
#include "Dynamic/FEM/ContinuousMaterial.h"
struct PBD_ContinuousMaterial : public ContinuousMaterial {
    PBD_ContinuousMaterial(const scalar _young, const scalar _poisson) : ContinuousMaterial(_young, _poisson) { }
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
        P = 2.f * trace * Matrix::Identity3x3();
    }

    virtual scalar getStiffness() const override { return this->lambda; }
};

struct Hooke_Second : public PBD_ContinuousMaterial {

    Hooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override {
        const auto E = getStrainTensorLinear(F);
        // P(F) = 2E
        // C(F) = tr(E²)
        P = 2.f * E;
        energy = Matrix::SquaredTrace(E);
    }

    virtual scalar getStiffness() const override { return this->mu * 2.f; }
};

struct StVK_First : public PBD_ContinuousMaterial {
    StVK_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override {
        const auto trace = Matrix::Trace(getStrainTensor(F));
        // P(F) = 2 tr(E) F 
        P = 2.f * trace * F;

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
        P =  4.f * F * E;
        // C(F) = tr(E^2)
        energy = 2.f * Matrix::SquaredTrace(E);
    }
    virtual scalar getStiffness() const override { return this->mu; }
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
        scalar shift = this->mu / this->lambda;
        energy = (detF) * (detF);
        Matrix3x3 d_detF; // derivative of det(F) by F
        d_detF[0] = glm::cross(F[1], F[2]);
        d_detF[1] = glm::cross(F[2], F[0]);
        d_detF[2] = glm::cross(F[0], F[1]);
        P = 2.f * detF * d_detF;
    }

    virtual scalar getStiffness() const override { return this->lambda; }
};

struct Stable_NeoHooke_First : public VolumePreservation {
    Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : VolumePreservation(_young, _poisson) {
        this->alpha = scalar(1) + this->mu / this->lambda;
    }
};

struct Stable_NeoHooke_Second : public PBD_ContinuousMaterial {
    Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    virtual void getStressTensorAndEnergy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override
    {
        energy = Matrix::SquaredNorm(F) - 3.f;

        P = 2.f * F;
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
        P = 2.f * F - 2.f * d_detF;
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

        P = 4.f * IVc_1 * glm::outerProduct(Fa, a);
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

        energy = (dFa - 1.f) * (dFa - 1.f);

        P = 2.f * (1.f - 1.f / dFa) * glm::outerProduct(Fa, a);
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
        energy = glm::determinant(F) - 1.f;
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
        energy = Matrix::SquaredNorm(F) - 3.f - 2.f * (glm::determinant(F) - 1.f);
        energy = sqrt(abs(energy));

        Matrix3x3 d_det;
        d_det[0] = glm::cross(F[1], F[2]);
        d_det[1] = glm::cross(F[2], F[0]);
        d_det[2] = glm::cross(F[0], F[1]);
        P = (2.f * F - 2.f * d_det) * (1.f / (2.f*energy));
    }

    virtual scalar getStiffness() const override { return this->mu; }
};


std::vector<PBD_ContinuousMaterial*> get_pbd_materials(Material material, scalar young, scalar poisson) {
    std::vector<PBD_ContinuousMaterial*> materials;
    switch (material)
    {
    case Hooke:
        materials.push_back(new Hooke_First(young, poisson));
        materials.push_back(new Hooke_Second(young, poisson));
        break;
    case StVK:
        materials.push_back(new StVK_First(young, poisson));
        materials.push_back(new StVK_Second(young, poisson));
        break;
    case Neo_Hooke:
        materials.push_back(new Stable_NeoHooke_First(young, poisson));
        materials.push_back(new Stable_NeoHooke_Second(young, poisson));
        break;
    case Developed_Neohooke:
        materials.push_back(new Developed_Stable_NeoHooke_First(young, poisson));
        materials.push_back(new Developed_Stable_NeoHooke_Second(young, poisson));
        break;
    default:
        std::cout << "Material not found" << std::endl;
        break;
    }
    return materials;
}