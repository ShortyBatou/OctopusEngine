#pragma once
#include <vector>

#include "Dynamic/FEM/ContinuousMaterial.h"
struct PBD_ContinuousMaterial : ContinuousMaterial {
    explicit PBD_ContinuousMaterial(const scalar _young, const scalar _poisson) : ContinuousMaterial(_young, _poisson) { }
    // constraint version without the stiffness component
    virtual void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) = 0;

    [[nodiscard]] virtual scalar get_stiffness() const = 0;
    [[nodiscard]] scalar get_energy(const Matrix3x3& F) override;
    [[nodiscard]] Matrix3x3 get_pk1(const Matrix3x3& F) override;

    ~PBD_ContinuousMaterial() override = default;
};

struct Hooke_First final : PBD_ContinuousMaterial {
    explicit Hooke_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->lambda; }
};

struct Hooke_Second final : PBD_ContinuousMaterial {

    explicit Hooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu * 2.f; }
};

struct StVK_First final : PBD_ContinuousMaterial {
    explicit StVK_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->lambda; }
};

struct StVK_Second final : PBD_ContinuousMaterial {

    explicit StVK_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;
    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};

struct VolumePreservation : PBD_ContinuousMaterial {

    scalar alpha;

    explicit VolumePreservation(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson), alpha(1) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->lambda; }
};

struct Stable_NeoHooke_First final : VolumePreservation {
    explicit Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : VolumePreservation(_young, _poisson) {
        this->alpha = 1 + this->mu / this->lambda;
    }
};

struct Stable_NeoHooke_Second final : PBD_ContinuousMaterial {
    explicit Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};


struct NeoHooke_ln_First final : VolumePreservation {
    explicit NeoHooke_ln_First(const scalar _young, const scalar _poisson) : VolumePreservation(_young, _poisson) {
        alpha = 1 + mu / lambda;
    }
    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;
};

struct NeoHooke_ln_Second final : PBD_ContinuousMaterial {
    explicit NeoHooke_ln_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};


struct Developed_Stable_NeoHooke_First final : VolumePreservation {
    explicit Developed_Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : VolumePreservation(_young, _poisson) { }
};


struct Developed_Stable_NeoHooke_Second final : PBD_ContinuousMaterial {
    explicit Developed_Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};


struct Anisotropic final : PBD_ContinuousMaterial {
    Vector3 a;
    explicit Anisotropic(const scalar _young, const scalar _poisson, const Vector3& _a) : PBD_ContinuousMaterial(_young, _poisson), a(_a) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};


struct Sqrt_Anisotropic final : PBD_ContinuousMaterial {
    Vector3 a;
    explicit Sqrt_Anisotropic(const scalar _young, const scalar _poisson, const Vector3& _a) : PBD_ContinuousMaterial(_young, _poisson), a(_a) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};

// Combine multiple energies that have the same stifness factor
struct Material_Union final : PBD_ContinuousMaterial {
    std::vector<PBD_ContinuousMaterial*> materials;

    explicit Material_Union(const std::vector<PBD_ContinuousMaterial*> &_materials) : PBD_ContinuousMaterial(_materials[0]->young, _materials[0]->poisson), materials(_materials) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return materials[0]->get_stiffness(); }
    ~Material_Union() override {
        for (PBD_ContinuousMaterial* m : materials) delete m;
        materials.clear();
    }

};

//Muller and Macklin neohooke energy for Tetra
struct C_Stable_NeoHooke_First final : PBD_ContinuousMaterial {
    scalar alpha;

    C_Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) {
        alpha = 1 + mu / lambda;
    }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->lambda; }
};


struct C_Stable_NeoHooke_Second final : PBD_ContinuousMaterial {
    C_Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};


struct C_Developed_Stable_NeoHooke_First final : PBD_ContinuousMaterial {
    C_Developed_Stable_NeoHooke_First(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->lambda; }
};


struct C_Developed_Stable_NeoHooke_Second final : PBD_ContinuousMaterial {
    C_Developed_Stable_NeoHooke_Second(const scalar _young, const scalar _poisson) : PBD_ContinuousMaterial(_young, _poisson) { }

    void get_pk1_and_energy(const Matrix3x3& F, Matrix3x3& P, scalar& energy) override;

    [[nodiscard]] scalar get_stiffness() const override { return this->mu; }
};


std::vector<PBD_ContinuousMaterial*> get_pbd_materials(Material material, scalar young, scalar poisson);