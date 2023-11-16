#pragma once
#include <vector>
#include "Core/Base.h"

struct FEM_Shape {

    unsigned int nb;
    FEM_Shape(unsigned int _nb) : nb(_nb) {}
    virtual std::vector<scalar> getQuadratureCoordinates() const = 0;
    virtual std::vector<scalar> getWeights() const = 0;
    virtual std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const = 0;

    virtual std::vector<Vector3> convert_dN_to_vector3(scalar* dN) const {
        std::vector<Vector3> dN_v3(nb);
        for (unsigned int i = 0; i < nb; ++i) {
            dN_v3[i].x = dN[i];
            dN_v3[i].y = dN[i + nb];
            dN_v3[i].z = dN[i + nb * 2];
        }
        return dN_v3;
    }

    virtual ~FEM_Shape() {}
};

struct Tetra_4 : public FEM_Shape {

    Tetra_4() : FEM_Shape(4) {}

    virtual std::vector<scalar> getQuadratureCoordinates() const {
        return std::vector<scalar>(3, 0.25); // change nothing because it's a constant strain element
    }

    virtual std::vector<scalar> getWeights() const {
        return std::vector<scalar>(1, 1. / 6.);
    }

    virtual std::vector<Vector3> build_shape_derivatives(scalar, scalar, scalar) const {
        scalar dN[4 * 3] = {
            -1, 1, 0, 0,
            -1, 0, 1, 0,
            -1, 0, 0, 1 
        };

        return this->convert_dN_to_vector3(dN);
    }
};

struct Pyramid_5 : public FEM_Shape {

    Pyramid_5() :FEM_Shape(5) { }

    virtual std::vector<scalar> getQuadratureCoordinates() const {
        scalar h1 = 0.1331754163448146;
        scalar h2 = 0.6372983346207416;
        scalar a = 0.5;
        return { a,0,h1, 0,a,h1, -a,0,h1, 0,-a,h1, 0,0,h2 };
    }

    virtual std::vector<scalar> getWeights() const {
        return std::vector<scalar>(5, 2. / 15.);
    }

    virtual std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const {
        scalar dN[5*3] = {
                -(l - s - t - 1) / (4 - 4 * l) - (l - s + t - 1) / (4 - 4 * l),
                (l - s - t - 1) / (4 - 4 * l) - (l + s - t - 1) / (4 - 4 * l),
                (l + s - t - 1) / (4 - 4 * l) + (l + s + t - 1) / (4 - 4 * l),
                (l - s + t - 1) / (4 - 4 * l) - (l + s + t - 1) / (4 - 4 * l),
                0,
            
            
                (l - s - t - 1) / (4 - 4 * l) - (l - s + t - 1) / (4 - 4 * l),
                -(l - s - t - 1) / (4 - 4 * l) - (l + s - t - 1) / (4 - 4 * l),
                (l + s - t - 1) / (4 - 4 * l) - (l + s + t - 1) / (4 - 4 * l),
                (l - s + t - 1) / (4 - 4 * l) + (l + s + t - 1) / (4 - 4 * l),
                0,
            
            
                (l - s - t - 1) / (4 - 4 * l) + (l - s + t - 1) / (4 - 4 * l) + 4 * (l - s - t - 1) * (l - s + t - 1) / ((4 - 4 * l) * (4 - 4 * l)),
                (l - s - t - 1) / (4 - 4 * l) + (l + s - t - 1) / (4 - 4 * l) + 4 * (l - s - t - 1) * (l + s - t - 1) / ((4 - 4 * l) * (4 - 4 * l)),
                (l + s - t - 1) / (4 - 4 * l) + (l + s + t - 1) / (4 - 4 * l) + 4 * (l + s - t - 1) * (l + s + t - 1) / ((4 - 4 * l) * (4 - 4 * l)),
                (l - s + t - 1) / (4 - 4 * l) + (l + s + t - 1) / (4 - 4 * l) + 4 * (l - s + t - 1) * (l + s + t - 1) / ((4 - 4 * l) * (4 - 4 * l)),
                1
            
        };

        return this->convert_dN_to_vector3(dN);
    }
};

struct Prysm_6 : public FEM_Shape {

    Prysm_6() : FEM_Shape(6) { }

    virtual std::vector<scalar> getQuadratureCoordinates() const {
        scalar a = 1. / std::sqrt(3.);
        scalar b = 0.5;
        return { b,0,-a, 0,b,-a, b,b,-a, b,0,a, 0,b,a, b,b,a };
    }

    virtual std::vector<scalar> getWeights() const {
        return std::vector<scalar>(6, 1. / 6.);
    }

    virtual std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const {
        scalar dN[6*3] {
            l - 1, 1 - l, 0, -l - 1, l + 1, 0,
            l - 1, 0, 1 - l, -l - 1, 0, l + 1,
            s + t - 1, -s, -t, -s - t + 1, s, t
        };

        for (unsigned int i = 0; i < 18; ++i) dN[i] *= 0.5;
        return this->convert_dN_to_vector3(dN);
    }
};


struct Hexa_8 : public FEM_Shape {

    Hexa_8() : FEM_Shape(8) {}

    virtual std::vector<scalar> getQuadratureCoordinates() const override {
        scalar c = 1. / std::sqrt(3);
        std::vector<scalar> coords = { -c,-c,-c, c,-c,-c, c,c,-c, -c,c,-c, -c,-c,c, c,-c,c, c,c,c, -c,c,c };
        return coords;
    }

    virtual std::vector<scalar> getWeights() const override {
        return std::vector<scalar>(8, 1.);
    }

    virtual std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const override {
        scalar dN[24] = {
            -(1 - l) * (1 - t), (1 - l)* (1 - t), (1 - l)* (t + 1), -(1 - l) * (t + 1), -(1 - t) * (l + 1), (1 - t)* (l + 1), (l + 1)* (t + 1), -(l + 1) * (t + 1),
            -(1 - l) * (1 - s), -(1 - l) * (s + 1), (1 - l)* (s + 1), (1 - l)* (1 - s), -(1 - s) * (l + 1), -(l + 1) * (s + 1), (l + 1)* (s + 1), (1 - s)* (l + 1),
            -(1 - s) * (1 - t), -(1 - t) * (s + 1), -(s + 1) * (t + 1), -(1 - s) * (t + 1), (1 - s)* (1 - t), (1 - t)* (s + 1), (s + 1)* (t + 1), (1 - s)* (t + 1)
        };

        for (unsigned int i = 0; i < 24; ++i) dN[i] *= scalar(1. / 8.);
        return this->convert_dN_to_vector3(dN);
    }
};


       

       
       
