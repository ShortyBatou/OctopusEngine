#pragma once
#include <Manager/Debug.h>

#include "Core/Base.h"
#include "Tools/Axis.h"

struct Area {
   virtual bool inside(const Vector3& p) const = 0;
   virtual void draw() const = 0;
   virtual ~Area() = default;
};

struct Box final : Area {
   Box(const Vector3& _pmin, const Vector3& _pmax) : pmin(_pmin), pmax(_pmax) {}
   ~Box() override = default;

   bool inside(const Vector3& p) const override {
      return   p.x >= pmin.x && p.y >= pmin.y && p.z >= pmin.z &&
               p.x <= pmax.x && p.y <= pmax.y && p.z <= pmax.z;
   }

   void draw() const override {
      Debug::Cube(pmin, pmax);
   }

   Vector3 pmin, pmax;
};


struct Plane final : Area {
   Plane(const Vector3& _o, const Vector3& _n) : o(_o), n(_n) {}
   bool inside(const Vector3& p) const override {
      return glm::dot(p - o, n) > 0;
   }

   void draw() const override {
      Debug::Line(o, o+n);
      Debug::Cube(o, 0.05);
   }

   ~Plane() override = default;
   Vector3 o, n;
};

struct Sphere final : Area {
   Sphere(const Vector3& _o, const scalar& _r) : o(_o), r(_r) {}
   bool inside(const Vector3& p) const override {
      return glm::length2(p-o) < r*r;
   }

   void draw() const override {
      Debug::Line(o, o + Unit3D::right() * r);
      Debug::Line(o, o + Unit3D::up() * r);
      Debug::Line(o, o + Unit3D::forward() * r);
   }

   ~Sphere() override = default;
   Vector3 o;
   scalar r;
};
