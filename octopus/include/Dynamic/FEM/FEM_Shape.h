#pragma once
#include <vector>
#include <Mesh/Mesh.h>
#include "Core/Base.h"

struct FEM_Shape {
    int nb; // number of vertices
    std::vector<std::vector<Vector3> > dN;
    std::vector<scalar> weights;

    explicit FEM_Shape(const int _nb) : nb(_nb), dN({}), weights({}) {
    }

    virtual void build();

    [[nodiscard]] virtual std::vector<scalar> get_quadrature_coordinates() const = 0;

    [[nodiscard]] virtual std::vector<scalar> get_weights() const = 0;

    [[nodiscard]] virtual Mesh::Geometry get_vertices() const = 0;

    [[nodiscard]] virtual std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const = 0;

    virtual std::vector<scalar> build_shape(scalar s, scalar t, scalar l) const = 0;

    virtual std::vector<Vector3> convert_dN_to_vector3(scalar *dN) const;

    virtual void debug_draw(std::vector<Vector3> &pts) {
    }

    virtual ~FEM_Shape() = default;
};


struct Tetra_4 final : FEM_Shape {
    Tetra_4() : FEM_Shape(4) {
        FEM_Shape::build();
    }

    [[nodiscard]] std::vector<scalar> get_quadrature_coordinates() const override {
        return std::vector(3, 0.25f); // change nothing because it's a constant strain element
    }

    [[nodiscard]] std::vector<scalar> get_weights() const override {
        return std::vector(1, 1.f / 6.f);
    }

    [[nodiscard]] std::vector<scalar> build_shape(scalar s, scalar t, scalar l) const override {
        return {1 - s - t - l, s, t, l};
    }

    [[nodiscard]] Mesh::Geometry get_vertices() const override {
        return {Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)};
    }

    [[nodiscard]] std::vector<Vector3> build_shape_derivatives(scalar, scalar, scalar) const override {
        scalar dN[4 * 3] = {
            -1, 1, 0, 0,
            -1, 0, 1, 0,
            -1, 0, 0, 1
        };

        return this->convert_dN_to_vector3(dN);
    }
};

struct Pyramid_5 final : FEM_Shape {
    Pyramid_5() : FEM_Shape(5) { FEM_Shape::build(); }

    [[nodiscard]] std::vector<scalar> get_quadrature_coordinates() const override {
        scalar h1 = 0.1331754163448146f;
        scalar h2 = 0.6372983346207416f;
        scalar a = 0.5f;
        return {a, 0, h1, 0, a, h1, -a, 0, h1, 0, -a, h1, 0, 0, h2};
    }

    [[nodiscard]] std::vector<scalar> get_weights() const override {
        return std::vector<scalar>(5, 2.f / 15.f);
    }

    [[nodiscard]] Mesh::Geometry get_vertices() const override {
        return {
            Vector3(-1, 0, -1), Vector3(1, 0, -1), Vector3(1, 0, 1), Vector3(-1, 0, -1), Vector3(0, 1, 0)
        }; // not sure
    }

    [[nodiscard]] std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const override {
        scalar dN[5 * 3] = {
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


            (l - s - t - 1) / (4 - 4 * l) + (l - s + t - 1) / (4 - 4 * l) + 4 * (l - s - t - 1) * (l - s + t - 1) / (
                (4 - 4 * l) * (4 - 4 * l)),
            (l - s - t - 1) / (4 - 4 * l) + (l + s - t - 1) / (4 - 4 * l) + 4 * (l - s - t - 1) * (l + s - t - 1) / (
                (4 - 4 * l) * (4 - 4 * l)),
            (l + s - t - 1) / (4 - 4 * l) + (l + s + t - 1) / (4 - 4 * l) + 4 * (l + s - t - 1) * (l + s + t - 1) / (
                (4 - 4 * l) * (4 - 4 * l)),
            (l - s + t - 1) / (4 - 4 * l) + (l + s + t - 1) / (4 - 4 * l) + 4 * (l - s + t - 1) * (l + s + t - 1) / (
                (4 - 4 * l) * (4 - 4 * l)),
            1

        };


        return this->convert_dN_to_vector3(dN);
    }

    [[nodiscard]] std::vector<scalar> build_shape(scalar s, scalar t, scalar l) const override {
        return {
            (-s + t + l - 1) * (-s - t + l - 1) / (4 * (1 - l)),
            (-s - t + l - 1) * (s - t + l - 1) / (4 * (1 - l)),
            (s + t + l - 1) * (s - t + l - 1) / (4 * (1 - l)),
            (s + t + l - 1) * (-s + t + l - 1) / (4 * (1 - l)),
            l,
        };
    }
};

struct Prism_6 final : FEM_Shape {
    Prism_6() : FEM_Shape(6) { FEM_Shape::build(); }

    [[nodiscard]] std::vector<scalar> get_quadrature_coordinates() const override {
        scalar a = 1.f / std::sqrt(3.f);
        scalar b = 0.5f;
        return {b, 0, -a, 0, b, -a, b, b, -a, b, 0, a, 0, b, a, b, b, a};
    }

    [[nodiscard]] std::vector<scalar> get_weights() const override {
        return std::vector(6, 1.f / 6.f);
    }

    [[nodiscard]] Mesh::Geometry get_vertices() const override {
        return {
            Vector3(0, -1, 0), Vector3(1, -1, 0), Vector3(0, -1, 1), Vector3(0, 1, 0), Vector3(1, 1, 0),
            Vector3(0, 1, 1)
        }; // not sure
    }

    [[nodiscard]] std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const override {
        scalar dN[6 * 3]{
            l - 1, 1 - l, 0, -l - 1, l + 1, 0,
            l - 1, 0, 1 - l, -l - 1, 0, l + 1,
            s + t - 1, -s, -t, -s - t + 1, s, t
        };

        for (float &i: dN) i *= 0.5f;
        return this->convert_dN_to_vector3(dN);
    }

    [[nodiscard]] std::vector<scalar> build_shape(scalar s, scalar t, scalar l) const override {
        return {
            0.5f * (1 - s - t) * (1 - l),
            0.5f * s * (1 - l),
            0.5f * t * (1 - l),
            0.5f * (1 - s - t) * (1 + l),
            0.5f * s * (1 + l),
            0.5f * t * (1 + l),
        };
    }
};


struct Hexa_8 final : FEM_Shape {
    Hexa_8() : FEM_Shape(8) { FEM_Shape::build(); }

    [[nodiscard]] std::vector<scalar> get_quadrature_coordinates() const override {
        scalar c = 1.f / std::sqrt(3.f);
        std::vector coords = {
            -c, -c, -c, c, -c, -c, c, c, -c, -c, c, -c, -c, -c, c, c, -c, c, c, c, c, -c, c, c
        };
        return coords;
    }

    [[nodiscard]] std::vector<scalar> get_weights() const override {
        return std::vector(8, 1.f);
    }

    [[nodiscard]] Mesh::Geometry get_vertices() const override {
        return {
            Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(1, 1, -1), Vector3(-1, 1, -1),
            Vector3(-1, -1, 1), Vector3(1, -1, 1), Vector3(1, 1, 1), Vector3(-1, 1, 1)
        };
    }

    [[nodiscard]] std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const override {
        scalar dN[24] = {
            -(1 - l) * (1 - t), (1 - l) * (1 - t), (1 - l) * (t + 1), -(1 - l) * (t + 1), -(1 - t) * (l + 1),
            (1 - t) * (l + 1), (l + 1) * (t + 1), -(l + 1) * (t + 1),
            -(1 - l) * (1 - s), -(1 - l) * (s + 1), (1 - l) * (s + 1), (1 - l) * (1 - s), -(1 - s) * (l + 1),
            -(l + 1) * (s + 1), (l + 1) * (s + 1), (1 - s) * (l + 1),
            -(1 - s) * (1 - t), -(1 - t) * (s + 1), -(s + 1) * (t + 1), -(1 - s) * (t + 1), (1 - s) * (1 - t),
            (1 - t) * (s + 1), (s + 1) * (t + 1), (1 - s) * (t + 1)
        };

        for (float &i: dN) i *= 1.f / 8.f;
        return this->convert_dN_to_vector3(dN);
    }

    [[nodiscard]] std::vector<scalar> build_shape(scalar s, scalar t, scalar l) const override {
        const scalar a = 1.f / 8.f;
        return {
            a * (1 - s) * (1 - t) * (1 - l),
            a * (1 + s) * (1 - t) * (1 - l),
            a * (1 + s) * (1 + t) * (1 - l),
            a * (1 - s) * (1 + t) * (1 - l),
            a * (1 - s) * (1 - t) * (1 + l),
            a * (1 + s) * (1 - t) * (1 + l),
            a * (1 + s) * (1 + t) * (1 + l),
            a * (1 - s) * (1 + t) * (1 + l)
        };
    }
};


struct Tetra_10 final : FEM_Shape {
    Tetra_10() : FEM_Shape(10) { FEM_Shape::build(); }

    [[nodiscard]] std::vector<scalar> get_quadrature_coordinates() const override {
        return {
            0.162001491698524457962804490307462401688098907470703125f,
            0.18385035049209774715706089409650303423404693603515625f,
            0.012718366313681450652239846021984703838825225830078125f,

            0.01090521221118924410919959200327866710722446441650390625f,
            0.2815238021235462184677089680917561054229736328125f,
            0.3621268299455338013359551041503436863422393798828125f,

            0.190117002439283921955137657278100959956645965576171875f,
            0.0114033294445571690978180612319192732684314250946044921875f,
            0.3586207204668838688377263679285533726215362548828125f,

            0.1708169251649890030275713570517837069928646087646484375f,
            0.15281814309092733861206170331570319831371307373046875f,
            0.63849329996172665691034353585564531385898590087890625f,

            0.158685163227440584332583739524125121533870697021484375f,
            0.5856628056552157790548562843468971550464630126953125f,
            0.130847168952096470917467740946449339389801025390625f,

            0.57122605214911514881492848871857859194278717041015625f,
            0.1469183900871695869216893015618552453815937042236328125f,
            0.1403728057942107143585275252917199395596981048583984375f
        };
    }

    [[nodiscard]] std::vector<scalar> get_weights() const override {
        //return std::vector<scalar>(4, scalar(1. / 4. * 1. / 6.));
        return
        {
            0.12232200275734507466385281304610543884336948394775390625f / 6.f,
            0.12806641271074692411957585136406123638153076171875f / 6.f,
            0.1325680271444452384965728697352460585534572601318359375f / 6.f,
            0.14062440966040323786501176073215901851654052734375f / 6.f,
            0.224415166917557418191364604354021139442920684814453125f / 6.f,
            0.252003980809502314830439217985258437693119049072265625f / 6.f
        };
    }

    [[nodiscard]] Mesh::Geometry get_vertices() const override {
        return {
            Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1),
            Vector3(0.5, 0, 0), Vector3(0.5, 0.5, 0.), Vector3(0., 0.5, 0.),
            Vector3(0, 0, 0.5), Vector3(0.5, 0., 0.5), Vector3(0., 0.5, 0.5)
        };
    }

    [[nodiscard]] std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const override {
        scalar dN[30] = {
            4 * l + 4 * s + 4 * t - 3, 4 * s - 1, 0, 0, -4 * l - 8 * s - 4 * t + 4, 4 * t, -4 * t, -4 * l, 4 * l, 0,
            4 * l + 4 * s + 4 * t - 3, 0, 4 * t - 1, 0, -4 * s, 4 * s, -4 * l - 4 * s - 8 * t + 4, -4 * l, 0, 4 * l,
            4 * l + 4 * s + 4 * t - 3, 0, 0, 4 * l - 1, -4 * s, 0, -4 * t, -8 * l - 4 * s - 4 * t + 4, 4 * s, 4 * t
        };
        return this->convert_dN_to_vector3(dN);
    }

    // compute N_i(X)
    [[nodiscard]] std::vector<scalar> build_shape(scalar s, scalar t, scalar l) const override {
        return {
            (1 - s - t - l) * (2.f * (1 - s - t - l) - 1),
            s * (2 * s - 1),
            t * (2 * t - 1),
            l * (2 * l - 1),
            4 * s * (1 - s - t - l),
            4 * s * t,
            4 * t * (1 - s - t - l),
            4 * l * (1 - s - t - l),
            4 * s * l,
            4 * t * l
        };
    }
};


struct Tetra_20 final : FEM_Shape {
    Tetra_20() : FEM_Shape(20) { FEM_Shape::build(); }

    [[nodiscard]] std::vector<scalar> get_quadrature_coordinates() const override {
        // polyfem
        std::vector<scalar> coords = {
            0.04049050672759042790449512949635391123592853546142578125f,
            0.0135607018798028812478495552795720868743956089019775390625f,
            0.77125473269537614395829905333812348544597625732421875f,
            0.75250850700965499218142440440715290606021881103515625f,
            0.06809937093820665754417831294631469063460826873779296875f,
            0.09798720364927911152808093220301088877022266387939453125f,
            0.06722329489338339791881793416905566118657588958740234375f,
            0.0351839297735987155402170856177690438926219940185546875f,
            0.1563638932393952851729324038387858308851718902587890625f,
            0.419266313879513019546863006326020695269107818603515625f,
            0.04778143555908666295639619647772633470594882965087890625f,
            0.4796110110256550651541829211055301129817962646484375f,
            0.45076587609127682920728830140433274209499359130859375f,
            0.059456616299433828753961961410823278129100799560546875f,
            0.056824017127933668103167974550160579383373260498046875f,
            0.1294113737889104054357147788323345594108104705810546875f,
            0.33019041483746447429581394317210651934146881103515625f,
            0.00239100745743936471399138099513947963714599609375f,
            0.12154199133392780407536548636926454491913318634033203125f,
            0.306493988429690278341155362795689143240451812744140625f,
            0.562972760143046091485530268982984125614166259765625f,
            0.09720464458758326653509129755548201501369476318359375f,
            0.684390415453040024118536166497506201267242431640625f,
            0.111800767397383093992857538978569209575653076171875f,
            0.029569495206479612381400556841981597244739532470703125f,
            0.317903560213394609235137977520935237407684326171875f,
            0.3232939848374789537643891890184022486209869384765625f,
            0.432710239047768563391827001396450214087963104248046875f,
            0.353823239209297091267814039383665658533573150634765625f,
            0.10962240533194123059956837096251547336578369140625f,
            0.2402766649280726196646895687081268988549709320068359375f,
            0.126801725915392016208471659410861320793628692626953125f,
            0.328473206722038446603306738325045444071292877197265625f
        };
        return coords;
    }

    [[nodiscard]] std::vector<scalar> get_weights() const override {
        // polyfem
        std::vector w = {
            0.03925109092483995698596999091023462824523448944091796875f / 6.f,
            0.055273369155936898089453990223773871548473834991455078125f / 6.f,
            0.055393798871576367670588325609060120768845081329345703125f / 6.f,
            0.05993318514655952833347640762440278194844722747802734375f / 6.f,
            0.06946996593763536675947278808962437324225902557373046875f / 6.f,
            0.07616271524555835725767138910669018514454364776611328125f / 6.f,
            0.0794266800680253071131886599687277339398860931396484375f / 6.f,
            0.10646803415549009608209729549344046972692012786865234375f / 6.f,
            0.11023423242849765546491624945701914839446544647216796875f / 6.f,
            0.1549761160162460849054610889652394689619541168212890625f / 6.f,
            0.193410812049634450726642853624070994555950164794921875f / 6.f
        };
        return w;
    }

    [[nodiscard]] Mesh::Geometry get_vertices() const override {
        const scalar a = 1.f / 3.f;
        const scalar b = 2.f / 3.f;

        return {
            Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1), // 0,1,2,3
            // edges
            Vector3(a, 0, 0), Vector3(b, 0, 0), // 4,5
            Vector3(b, a, 0), Vector3(a, b, 0), // 6,7
            Vector3(0, b, 0), Vector3(0, a, 0), // 8,9

            Vector3(0, 0, a), Vector3(0, 0, b), //10, 11

            Vector3(b, 0, a), Vector3(a, 0, b), // 12,13
            Vector3(0, b, a), Vector3(0, a, b), // 14, 15

            // faces
            Vector3(a, 0, a), // 16
            Vector3(a, a, a), // 17
            Vector3(0, a, a), // 18
            Vector3(a, a, 0), // 19
        };
    }


    [[nodiscard]] std::vector<Vector3> build_shape_derivatives(scalar x, scalar y, scalar z) const override {
        scalar dN[60] = {
            -(-3 * x - 3 * y - 3 * z + 1) * (-1.5f * x - 1.5f * y - 1.5f * z + 1.f) - 1.5f * (
                -3 * x - 3 * y - 3 * z + 1) *
            (-x - y - z + 1) - 3 * (-1.5f * x - 1.5f * y - 1.5f * z + 1.f) * (-x - y - z + 1),
            3 * x * (1.5f * x - 0.5f) + 1.5f * x * (3 * x - 2) + (1.5f * x - 0.5f) * (3 * x - 2),
            0,
            0,
            -3 * x * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * x * (-3 * x - 3 * y - 3 * z + 2) + (
                -4.5f * x - 4.5f * y - 4.5f * z + 4.5f) * (-3 * x - 3 * y - 3 * z + 2),
            -4.5f * x * (3 * x - 1) + 3 * x * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) + (3 * x - 1) * (
                -4.5f * x - 4.5f * y - 4.5f * z + 4.5f),
            13.5f * x * y + 4.5f * y * (3 * x - 1),
            4.5f * y * (3 * y - 1),
            -4.5f * y * (3 * y - 1),
            -3 * y * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * y * (-3 * x - 3 * y - 3 * z + 2),
            -3 * z * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * z * (-3 * x - 3 * y - 3 * z + 2),
            -4.5f * z * (3 * z - 1),
            13.5f * x * z + 4.5f * z * (3 * x - 1),
            4.5f * z * (3 * z - 1),
            0,
            0,
            -27.f * x * z + z * (-27.f * x - 27.f * y - 27.f * z + 27.f),
            27.f * y * z,
            -27.f * y * z,
            -27.f * x * y + y * (-27.f * x - 27.f * y - 27.f * z + 27.f),

            -(-3 * x - 3 * y - 3 * z + 1) * (-1.5f * x - 1.5f * y - 1.5f * z + 1.f) - 1.5f * (
                -3 * x - 3 * y - 3 * z + 1) *
            (-x - y - z + 1) - 3 * (-1.5f * x - 1.5f * y - 1.5f * z + 1.f) * (-x - y - z + 1),
            0,
            3 * y * (1.5f * y - 0.5f) + 1.5f * y * (3 * y - 2) + (1.5f * y - 0.5f) * (3 * y - 2),
            0,
            -3 * x * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * x * (-3 * x - 3 * y - 3 * z + 2),
            -4.5f * x * (3 * x - 1),
            4.5f * x * (3 * x - 1),
            13.5f * x * y + 4.5f * x * (3 * y - 1),
            -4.5f * y * (3 * y - 1) + 3 * y * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) + (3 * y - 1) * (
                -4.5f * x - 4.5f * y - 4.5f * z + 4.5f),
            -3 * y * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * y * (-3 * x - 3 * y - 3 * z + 2) + (
                -4.5f * x - 4.5f * y - 4.5f * z + 4.5f) * (-3 * x - 3 * y - 3 * z + 2),
            -3 * z * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * z * (-3 * x - 3 * y - 3 * z + 2),
            -4.5f * z * (3 * z - 1),
            0,
            0, 13.5f * y * z + 4.5f * z * (3 * y - 1),
            4.5f * z * (3 * z - 1),
            -27.f * x * z,
            27.f * x * z,
            -27.f * y * z + z * (-27.f * x - 27.f * y - 27.f * z + 27.f),
            -27.f * x * y + x * (-27.f * x - 27.f * y - 27.f * z + 27.f),
            -(-3 * x - 3 * y - 3 * z + 1) * (-1.5f * x - 1.5f * y - 1.5f * z + 1.f) - 1.5f * (
                -3 * x - 3 * y - 3 * z + 1) *
            (-x - y - z + 1) - 3 * (-1.5f * x - 1.5f * y - 1.5f * z + 1.f) * (-x - y - z + 1),
            0,
            0,
            3 * z * (1.5f * z - 0.5f) + 1.5f * z * (3 * z - 2) + (1.5f * z - 0.5f) * (3 * z - 2),
            -3 * x * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * x * (-3 * x - 3 * y - 3 * z + 2),
            -4.5f * x * (3 * x - 1),
            0,
            0,
            -4.5f * y * (3 * y - 1),
            -3 * y * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * y * (-3 * x - 3 * y - 3 * z + 2),
            -3 * z * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) - 4.5f * z * (-3 * x - 3 * y - 3 * z + 2) + (
                -4.5f * x - 4.5f * y - 4.5f * z + 4.5f) * (-3 * x - 3 * y - 3 * z + 2),
            -4.5f * z * (3 * z - 1) + 3 * z * (-4.5f * x - 4.5f * y - 4.5f * z + 4.5f) + (3 * z - 1) * (
                -4.5f * x - 4.5f * y - 4.5f * z + 4.5f),
            4.5f * x * (3 * x - 1),
            13.5f * x * z + 4.5f * x * (3 * z - 1),
            4.5f * y * (3 * y - 1),
            13.5f * y * z + 4.5f * y * (3 * z - 1),
            -27.f * x * z + x * (-27.f * x - 27.f * y - 27.f * z + 27.f),
            27.f * x * y,
            -27.f * y * z + y * (-27.f * x - 27.f * y - 27.f * z + 27.f),
            -27.f * x * y
        };
        return this->convert_dN_to_vector3(dN);
    }

    // compute N_i(X)
    [[nodiscard]] std::vector<scalar> build_shape(scalar x, scalar y, scalar z) const override {
        const scalar a = 0.5;
        const scalar b = 9. / 2.;
        const scalar c = 27.;
        return {
            //corner nodes
            a * (3 * (1 - x - y - z) - 1) * (3 * (1 - x - y - z) - 2) * (1 - x - y - z), // 0
            a * (3 * x - 1) * (3 * x - 2) * x, //1

            a * (3 * y - 1) * (3 * y - 2) * y, //2
            a * (3 * z - 1) * (3 * z - 2) * z, //3

            //mid edge nodes
            b * (1 - x - y - z) * x * (3 * (1 - x - y - z) - 1), // 4
            b * (1 - x - y - z) * x * (3 * x - 1), // 5

            b * x * y * (3 * x - 1), //6
            b * x * y * (3 * y - 1), //7

            b * (1 - x - y - z) * y * (3 * y - 1), //8
            b * (1 - x - y - z) * y * (3 * (1 - x - y - z) - 1), //9

            b * (1 - x - y - z) * z * (3 * (1 - x - y - z) - 1), //10
            b * (1 - x - y - z) * z * (3 * z - 1), //11
            b * x * z * (3 * x - 1), //12

            b * x * z * (3 * z - 1), //13

            b * y * z * (3 * y - 1), //14
            b * y * z * (3 * z - 1), //15

            //mid face nodes
            c * (1 - x - y - z) * x * z, // 16

            c * x * y * z, // 17
            c * (1 - x - y - z) * y * z, // 18
            c * (1 - x - y - z) * x * y //19
        };
    }

    void debug_draw(std::vector<Vector3> &pts) override {
    }
};

FEM_Shape *get_fem_shape(Element type);
