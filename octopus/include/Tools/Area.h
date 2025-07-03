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
   Box(const Mesh::Geometry& geometry) {
      assert(geometry.size() > 0);
      pmin = geometry[0];
      pmax = geometry[0];
      for(const Vector3& x : geometry) {
         pmin = glm::min(pmin, x);
         pmax = glm::max(pmax, x);
      }
   }

   Box(const Vector3* geometry, const int* topology, const int nb) {
      assert(nb > 0);
      pmin = geometry[topology[0]];
      pmax = geometry[topology[0]];
      for(int i = 1; i < nb; ++i) {
         pmin = glm::min(pmin, geometry[topology[i]]);
         pmax = glm::max(pmax, geometry[topology[i]]);
      }
   }

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


struct Tetraedron final : Area {
   Tetraedron(const Vector3& a, const Vector3& b, const Vector3& c, const Vector3& d) {
      _p[0] = a; _p[1] = b; _p[2] = c; _p[3] = d;
   }

   Tetraedron(const Vector3* p) {
      for(int i = 0; i < 4; ++i) _p[i] = p[i];
   }

   Tetraedron(const Vector3* geo, const int* topo) {
      for(int i = 0; i < 4; ++i) _p[i] = geo[topo[i]];
   }

   bool inside(const Vector3& p) const override {
      const scalar V0 = signed_volume();
      const bool b1 = V0 * Signed_Volume(p, _p[1], _p[2], _p[3]) > -eps;
      const bool b2 = V0 * Signed_Volume(_p[0], p, _p[2], _p[3]) > -eps;
      const bool b3 = V0 * Signed_Volume(_p[0], _p[1], p, _p[3]) > -eps;
      const bool b4 = V0 * Signed_Volume(_p[0], _p[1], _p[2], p) > -eps;
      if (b1 && b2 && b3 && b4) return true;
      return false;
   }

   Vector4 barycentric(const Vector3& p) const {
      const scalar inv_V0 = 1.f / signed_volume();
      const scalar l0 = Signed_Volume(p, _p[1], _p[2], _p[3]) * inv_V0;
      const scalar l1 = Signed_Volume(_p[0], p, _p[2], _p[3]) * inv_V0;
      const scalar l2 = Signed_Volume(_p[0], _p[1], p, _p[3]) * inv_V0;
      return Vector4(l0, l1, l2, 1.0 - l0 - l1 - l2);
   }

   scalar signed_volume() const {
      return glm::dot(_p[3] - _p[0], cross(_p[1] - _p[0], _p[2] - _p[0])) / 6.f;
   }

   static scalar Signed_Volume(const Vector3 a, const Vector3 b, const Vector3 c, const Vector3 d) {
      return glm::dot(d - a, cross(b - a, c - a)) / 6.f;
   }

   scalar volume() const {
      return abs(signed_volume());
   }

   void draw() const override {
      Debug::Line(_p[0], _p[1]);
      Debug::Line(_p[1], _p[2]);
      Debug::Line(_p[2], _p[0]);
      Debug::Line(_p[0], _p[3]);
      Debug::Line(_p[1], _p[3]);
      Debug::Line(_p[2], _p[3]);
   }

   ~Tetraedron() override = default;
   Vector3 _p[4];
};