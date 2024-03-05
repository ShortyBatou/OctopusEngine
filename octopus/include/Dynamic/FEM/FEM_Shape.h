#pragma once
#include <vector>
#include "Core/Base.h"
#include "Manager/Debug.h"



struct FEM_Shape {

    unsigned int nb;
    FEM_Shape(unsigned int _nb) : nb(_nb) {}
    virtual std::vector<scalar> get_quadrature_coordinates() const = 0;
    virtual std::vector<scalar> get_weights() const = 0;
    virtual Mesh::Geometry get_vertices() const = 0;
    virtual std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const = 0;
    virtual std::vector<scalar> build_shape(scalar s, scalar t, scalar l) = 0;
    virtual std::vector<Vector3> convert_dN_to_vector3(scalar* dN) const {
        std::vector<Vector3> dN_v3(nb);
        for (unsigned int i = 0; i < nb; ++i) {
            dN_v3[i].x = dN[i];
            dN_v3[i].y = dN[i + nb];
            dN_v3[i].z = dN[i + nb * 2];
        }
        return dN_v3;
    }

    virtual void debug_draw(std::vector<Vector3>& pts) { }

    virtual ~FEM_Shape() {}
};

struct Tetra_4 : public FEM_Shape {

    Tetra_4() : FEM_Shape(4) {}

    virtual std::vector<scalar> get_quadrature_coordinates() const {
        return std::vector<scalar>(3, 0.25); // change nothing because it's a constant strain element
    }

    virtual std::vector<scalar> get_weights() const {
        return std::vector<scalar>(1, 1. / 6.);
    }
    virtual std::vector<scalar> build_shape(scalar s, scalar t, scalar l) override {
        return { 1-s-t-l, s, t, l };
    }

    virtual Mesh::Geometry get_vertices() const override {
        return { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1) };
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

    virtual std::vector<scalar> get_quadrature_coordinates() const {
        scalar h1 = 0.1331754163448146;
        scalar h2 = 0.6372983346207416;
        scalar a = 0.5;
        return { a,0,h1, 0,a,h1, -a,0,h1, 0,-a,h1, 0,0,h2 };
    }

    virtual std::vector<scalar> get_weights() const {
        return std::vector<scalar>(5, 2. / 15.);
    }
    virtual Mesh::Geometry get_vertices() const {
        return { Vector3(-1, 0, -1), Vector3(1, 0, -1), Vector3(1, 0, 1), Vector3(-1, 0, -1), Vector3(0, 1, 0) }; // not sure
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

    virtual std::vector<scalar> build_shape(scalar s, scalar t, scalar l) override {
        return {
            (-s + t + l - 1) * (-s - t + l - 1) / (4 * (1 - l)),
            (-s - t + l - 1) * ( s - t + l - 1) / (4 * (1 - l)),
            ( s + t + l - 1) * ( s - t + l - 1) / (4 * (1 - l)),
            ( s + t + l - 1) * (-s + t + l - 1) / (4 * (1 - l)),
            l,
        };
    }
};

struct Prism_6 : public FEM_Shape {

    Prism_6() : FEM_Shape(6) { }

    virtual std::vector<scalar> get_quadrature_coordinates() const {
        scalar a = 1. / std::sqrt(3.);
        scalar b = 0.5;
        return { b,0,-a, 0,b,-a, b,b,-a, b,0,a, 0,b,a, b,b,a };
    }

    virtual std::vector<scalar> get_weights() const {
        return std::vector<scalar>(6, 1. / 6.);
    }

    virtual Mesh::Geometry get_vertices() const override {
        return { Vector3(0, -1, 0), Vector3(1, -1, 0), Vector3(0, -1, 1), Vector3(0, 1, 0), Vector3(1, 1, 0), Vector3(0, 1, 1) }; // not sure
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

    virtual std::vector<scalar> build_shape(scalar s, scalar t, scalar l) override {
        return {
            scalar(0.5) * (1 - s - t) * (1 - l),
            scalar(0.5) * s * (1 - l),
            scalar(0.5) * t * (1 - l),
            scalar(0.5) * (1 - s - t)* (1 + l),
            scalar(0.5) * s * (1 + l),
            scalar(0.5) * t * (1 + l),
        };
    }
};


struct Hexa_8 : public FEM_Shape {

    Hexa_8() : FEM_Shape(8) {}

    virtual std::vector<scalar> get_quadrature_coordinates() const override {
        scalar c = 1. / std::sqrt(3);
        std::vector<scalar> coords = { -c,-c,-c, c,-c,-c, c,c,-c, -c,c,-c, -c,-c,c, c,-c,c, c,c,c, -c,c,c };
        return coords;
    }

    virtual std::vector<scalar> get_weights() const override {
        return std::vector<scalar>(8, 1.);
    }

    virtual Mesh::Geometry get_vertices() const override {
        return { Vector3(-1, -1, -1), Vector3(1, -1, -1), Vector3(1, 1, -1), Vector3(-1, 1, -1),
                 Vector3(-1, -1,  1), Vector3(1,  -1, 1), Vector3(1, 1, 1), Vector3(-1,  1, 1)
        };
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

    virtual std::vector<scalar> build_shape(scalar s, scalar t, scalar l) override {
        const scalar a = scalar(1) / scalar(8);
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


struct Tetra_10 : public FEM_Shape {

    Tetra_10() : FEM_Shape(10) {}

    virtual std::vector<scalar> get_quadrature_coordinates() const override {
        return {
            0.162001491698524457962804490307462401688098907470703125,
            0.18385035049209774715706089409650303423404693603515625,
            0.012718366313681450652239846021984703838825225830078125,

            0.01090521221118924410919959200327866710722446441650390625,
            0.2815238021235462184677089680917561054229736328125,
            0.3621268299455338013359551041503436863422393798828125,

            0.190117002439283921955137657278100959956645965576171875,
            0.0114033294445571690978180612319192732684314250946044921875,
            0.3586207204668838688377263679285533726215362548828125,

            0.1708169251649890030275713570517837069928646087646484375,
            0.15281814309092733861206170331570319831371307373046875,
            0.63849329996172665691034353585564531385898590087890625,

            0.158685163227440584332583739524125121533870697021484375,
            0.5856628056552157790548562843468971550464630126953125,
            0.130847168952096470917467740946449339389801025390625,

            0.57122605214911514881492848871857859194278717041015625,
            0.1469183900871695869216893015618552453815937042236328125,
            0.1403728057942107143585275252917199395596981048583984375
        };
    
    }

    virtual std::vector<scalar> get_weights() const override {
        //return std::vector<scalar>(4, scalar(1. / 4. * 1. / 6.));
        return
        { 0.12232200275734507466385281304610543884336948394775390625 / 6.,
          0.12806641271074692411957585136406123638153076171875  / 6.,
          0.1325680271444452384965728697352460585534572601318359375 / 6.,
          0.14062440966040323786501176073215901851654052734375 / 6.,
          0.224415166917557418191364604354021139442920684814453125 / 6.,
          0.252003980809502314830439217985258437693119049072265625 / 6.
        }; 
    }

    virtual Mesh::Geometry get_vertices() const override {
        return { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1),
                 Vector3(0.5, 0, 0), Vector3(0.5, 0.5, 0.), Vector3(0., 0.5, 0.),
                 Vector3(0, 0, 0.5), Vector3(0.5, 0., 0.5),   Vector3(0., 0.5, 0.5)
        };

        
    }

    virtual std::vector<Vector3> build_shape_derivatives(scalar s, scalar t, scalar l) const override {
        scalar dN[30] = {
            4 * l + 4 * s + 4 * t - 3, 4 * s - 1, 0, 0, -4 * l - 8 * s - 4 * t + 4, 4 * t, -4 * t, -4 * l, 4 * l, 0,
            4 * l + 4 * s + 4 * t - 3, 0, 4 * t - 1, 0, -4 * s, 4 * s, -4 * l - 4 * s - 8 * t + 4, -4 * l, 0, 4 * l,
            4 * l + 4 * s + 4 * t - 3, 0, 0, 4 * l - 1, -4 * s, 0, -4 * t, -8 * l - 4 * s - 4 * t + 4, 4 * s, 4 * t
        };
        return this->convert_dN_to_vector3(dN);
    }

    // compute N_i(X) 
    virtual std::vector<scalar> build_shape(scalar s, scalar t, scalar l) override {
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

    void debug_draw(std::vector<Vector3>& pts) override {
        std::vector<Vector3> vertices = { Vector3(0,0,0), Vector3(1,0,0), Vector3(0,0,1), Vector3(0,1,0) };
        std::vector<unsigned int> edges = { 0,1,0,2,0,3,1,3,2,3,1,2 };
        unsigned int sub_dibivision = 8;
        scalar step = scalar(1) / scalar(sub_dibivision);

        Debug::SetColor(ColorBase::Black());
        
        for (unsigned int i = 0; i < edges.size(); i += 2) {
            scalar x = 0;
            for (unsigned int j = 0; j < sub_dibivision; ++j) {
                Vector3 a = Unit3D::Zero(), b = Unit3D::Zero();
                Vector3 p = vertices[edges[i]] * (1.f - x) + vertices[edges[i+1]] * x;
                std::vector<scalar> N = build_shape(p.x, p.y, p.z);
                for (unsigned int n = 0; n < N.size(); ++n) {
                    a += N[n] * pts[n];
                }
                // a = N_i(p) * pts[i]
            
                x += step;
                p = vertices[edges[i]] * (1.f - x) + vertices[edges[i+1]] * x;
                N = build_shape(p.x, p.y, p.z);
                for (unsigned int n = 0; n < N.size(); ++n) {
                    b += N[n] * pts[n];
                }
                // b = N_i(p) * pts[i]
                Debug::Line(a, b);
            }
        }
    }
};


struct Tetra_20 : public FEM_Shape {

    Tetra_20() : FEM_Shape(20) {}
    virtual std::vector<scalar> get_quadrature_coordinates() const override {
        // polyfem
        std::vector<scalar> coords ={ 
            0.04049050672759042790449512949635391123592853546142578125, 0.0135607018798028812478495552795720868743956089019775390625,  0.77125473269537614395829905333812348544597625732421875, 
            0.75250850700965499218142440440715290606021881103515625, 0.06809937093820665754417831294631469063460826873779296875, 0.09798720364927911152808093220301088877022266387939453125,
            0.06722329489338339791881793416905566118657588958740234375, 0.0351839297735987155402170856177690438926219940185546875, 0.1563638932393952851729324038387858308851718902587890625,
            0.419266313879513019546863006326020695269107818603515625, 0.04778143555908666295639619647772633470594882965087890625, 0.4796110110256550651541829211055301129817962646484375, 
            0.45076587609127682920728830140433274209499359130859375, 0.059456616299433828753961961410823278129100799560546875, 0.056824017127933668103167974550160579383373260498046875, 
            0.1294113737889104054357147788323345594108104705810546875, 0.33019041483746447429581394317210651934146881103515625, 0.00239100745743936471399138099513947963714599609375, 
            0.12154199133392780407536548636926454491913318634033203125, 0.306493988429690278341155362795689143240451812744140625, 0.562972760143046091485530268982984125614166259765625, 
            0.09720464458758326653509129755548201501369476318359375, 0.684390415453040024118536166497506201267242431640625, 0.111800767397383093992857538978569209575653076171875, 
            0.029569495206479612381400556841981597244739532470703125, 0.317903560213394609235137977520935237407684326171875, 0.3232939848374789537643891890184022486209869384765625, 
            0.432710239047768563391827001396450214087963104248046875, 0.353823239209297091267814039383665658533573150634765625, 0.10962240533194123059956837096251547336578369140625, 
            0.2402766649280726196646895687081268988549709320068359375, 0.126801725915392016208471659410861320793628692626953125, 0.328473206722038446603306738325045444071292877197265625 };
        return coords;
    }

    virtual std::vector<scalar> get_weights() const override {
        // polyfem
        std::vector<scalar> w = {
            0.03925109092483995698596999091023462824523448944091796875 / 6.,
            0.055273369155936898089453990223773871548473834991455078125 / 6.,
            0.055393798871576367670588325609060120768845081329345703125 / 6.,
            0.05993318514655952833347640762440278194844722747802734375 / 6.,
            0.06946996593763536675947278808962437324225902557373046875 / 6.,
            0.07616271524555835725767138910669018514454364776611328125 / 6.,
            0.0794266800680253071131886599687277339398860931396484375 / 6.,
            0.10646803415549009608209729549344046972692012786865234375 / 6.,
            0.11023423242849765546491624945701914839446544647216796875 / 6.,
            0.1549761160162460849054610889652394689619541168212890625 / 6.,
            0.193410812049634450726642853624070994555950164794921875 / 6.
        };
        return w;
    }

    virtual Mesh::Geometry get_vertices() const override {
        scalar a = scalar(1. / 3.);
        scalar b = scalar(2. / 3.);

        return { Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1), // 0,1,2,3
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


    virtual std::vector<Vector3> build_shape_derivatives(scalar x, scalar y, scalar z) const override {
        scalar dN[60] = {
            -(-3 * x - 3 * y - 3 * z + 1) * (-1.5 * x - 1.5 * y - 1.5 * z + 1.0) - 1.5 * (-3 * x - 3 * y - 3 * z + 1) * (-x - y - z + 1) - 3 * (-1.5 * x - 1.5 * y - 1.5 * z + 1.0) * (-x - y - z + 1), 
            3 * x * (1.5 * x - 0.5) + 1.5 * x * (3 * x - 2) + (1.5 * x - 0.5) * (3 * x - 2), 
            0, 
            0,
            -3 * x * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * x * (-3 * x - 3 * y - 3 * z + 2) + (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) * (-3 * x - 3 * y - 3 * z + 2), 
            -4.5 * x * (3 * x - 1) + 3 * x * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) + (3 * x - 1) * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5),
            13.5 * x * y + 4.5 * y * (3 * x - 1), 
            4.5 * y * (3 * y - 1), 
            -4.5 * y * (3 * y - 1), 
            -3 * y * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * y * (-3 * x - 3 * y - 3 * z + 2), 
            -3 * z * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * z * (-3 * x - 3 * y - 3 * z + 2), 
            -4.5 * z * (3 * z - 1), 
            13.5 * x * z + 4.5 * z * (3 * x - 1), 
            4.5 * z * (3 * z - 1), 
            0, 
            0, 
            -27.0 * x * z + z * (-27.0 * x - 27.0 * y - 27.0 * z + 27.0), 
            27.0 * y * z, 
            -27.0 * y * z, 
            -27.0 * x * y + y * (-27.0 * x - 27.0 * y - 27.0 * z + 27.0),
            
            -(-3 * x - 3 * y - 3 * z + 1) * (-1.5 * x - 1.5 * y - 1.5 * z + 1.0) - 1.5 * (-3 * x - 3 * y - 3 * z + 1) * (-x - y - z + 1) - 3 * (-1.5 * x - 1.5 * y - 1.5 * z + 1.0) * (-x - y - z + 1), 
            0, 
            3 * y * (1.5 * y - 0.5) + 1.5 * y * (3 * y - 2) + (1.5 * y - 0.5) * (3 * y - 2), 
            0, 
            -3 * x * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * x * (-3 * x - 3 * y - 3 * z + 2), 
            -4.5 * x * (3 * x - 1), 
            4.5 * x * (3 * x - 1), 
            13.5 * x * y + 4.5 * x * (3 * y - 1), 
            -4.5 * y * (3 * y - 1) + 3 * y * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) + (3 * y - 1) * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5), 
            -3 * y * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * y * (-3 * x - 3 * y - 3 * z + 2) + (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) * (-3 * x - 3 * y - 3 * z + 2), 
            -3 * z * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * z * (-3 * x - 3 * y - 3 * z + 2), 
            -4.5 * z * (3 * z - 1), 
            0, 
            0, 13.5 * y * z + 4.5 * z * (3 * y - 1), 
            4.5 * z * (3 * z - 1), 
            -27.0 * x * z, 
            27.0 * x * z, 
            -27.0 * y * z + z * (-27.0 * x - 27.0 * y - 27.0 * z + 27.0), 
            -27.0 * x * y + x * (-27.0 * x - 27.0 * y - 27.0 * z + 27.0),
            -(-3 * x - 3 * y - 3 * z + 1) * (-1.5 * x - 1.5 * y - 1.5 * z + 1.0) - 1.5 * (-3 * x - 3 * y - 3 * z + 1) * (-x - y - z + 1) - 3 * (-1.5 * x - 1.5 * y - 1.5 * z + 1.0) * (-x - y - z + 1), 
            0, 
            0, 
            3 * z * (1.5 * z - 0.5) + 1.5 * z * (3 * z - 2) + (1.5 * z - 0.5) * (3 * z - 2), 
            -3 * x * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * x * (-3 * x - 3 * y - 3 * z + 2), 
            -4.5 * x * (3 * x - 1), 
            0, 
            0, 
            -4.5 * y * (3 * y - 1), 
            -3 * y * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * y * (-3 * x - 3 * y - 3 * z + 2), 
            -3 * z * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) - 4.5 * z * (-3 * x - 3 * y - 3 * z + 2) + (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) * (-3 * x - 3 * y - 3 * z + 2), 
            -4.5 * z * (3 * z - 1) + 3 * z * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5) + (3 * z - 1) * (-4.5 * x - 4.5 * y - 4.5 * z + 4.5), 
            4.5 * x * (3 * x - 1), 
            13.5 * x * z + 4.5 * x * (3 * z - 1), 
            4.5 * y * (3 * y - 1),
            13.5 * y * z + 4.5 * y * (3 * z - 1), 
            -27.0 * x * z + x * (-27.0 * x - 27.0 * y - 27.0 * z + 27.0),
            27.0 * x * y, 
            -27.0 * y * z + y * (-27.0 * x - 27.0 * y - 27.0 * z + 27.0),
            -27.0 * x * y
        };
        return this->convert_dN_to_vector3(dN);
    }

    // compute N_i(X) 
    virtual std::vector<scalar> build_shape(scalar x, scalar y, scalar z) override {
        scalar a = scalar(0.5);
        scalar b = scalar(9. / 2.);
        scalar c = scalar(27.);
        return {
            //corner nodes
            a * (3 * (1 - x - y - z) - 1) * (3 * (1 - x - y - z) - 2) * (1 - x - y - z), // 0
            a * (3 * x - 1) * (3 * x - 2) * x,  //1
            
            a * (3 * y - 1) * (3 * y - 2) * y,  //2
            a * (3 * z - 1) * (3 * z - 2) * z,  //3

            //mid edge nodes
            b * (1 - x - y - z) * x * (3 * (1 - x - y - z) - 1), // 4
            b * (1 - x - y - z) * x * (3 * x - 1),    // 5

            b * x* y* (3 * x - 1), //6
            b * x* y* (3 * y - 1), //7
            
            b * (1 - x - y - z) * y * (3 * y - 1), //8
            b * (1 - x - y - z) * y * (3 * (1 - x - y - z) - 1), //9

            b * (1 - x - y - z) * z * (3 * (1 - x - y - z) - 1), //10
            b * (1 - x - y - z) * z * (3 * z - 1), //11
            b* x* z* (3 * x - 1), //12
            
            b * x * z * (3 * z - 1), //13

            b* y* z* (3 * y - 1), //14
            b * y * z * (3 * z - 1), //15 

            //mid face nodes
            c * (1 - x - y - z) * x * z, // 16

            c * x * y * z, // 17
            c * (1 - x - y - z) * y * z, // 18
            c * (1 - x - y - z) * x * y  //19
        };
    }

    void debug_draw(std::vector<Vector3>& pts) override {
  
    }
};

FEM_Shape* get_fem_shape(Element type) {
    FEM_Shape* shape;
    switch (type) {
    case Tetra: return new Tetra_4(); break;
    case Pyramid: return new Pyramid_5(); break;
    case Prism: return new Prism_6(); break;
    case Hexa: return new Hexa_8(); break;
    case Tetra10: return new Tetra_10(); break;
    case Tetra20: return new Tetra_20(); break;
    default: std::cout << "build_element : element not found " << type << std::endl; return nullptr;
    }
}

       
       
