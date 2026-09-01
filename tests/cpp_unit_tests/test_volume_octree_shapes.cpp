//! Cross-section kernels: disk-against-polygon area, and the diagonal-quadric
//! generalisation of the revolved-surface profile.
//!
//!   g++ -std=c++17 -O2 -I include -I <fmt> \
//!       tests/cpp_unit_tests/test_volume_octree_shapes.cpp src/position.cpp -o
//!       t
//!
//! The strongest checks here are cross-checks against already-verified code
//! (circle_poly_area must reproduce circle_rect_area on rectangles) and exact
//! identities (splitting a polygon must split the area), rather than numerical
//! references.

#include "openmc/volume_octree_math.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <random>

using namespace openmc;
static int fails = 0;
static std::mt19937 rng(99);
static double U(double a, double b)
{
  return std::uniform_real_distribution<double>(a, b)(rng);
}
#define CK(c, ...)                                                             \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("FAIL ");                                                         \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      ++fails;                                                                 \
    }                                                                          \
  } while (0)

static void check(const char* name, QuadricForm q, OctBox b, double exact)
{
  AxisSymProfile p;
  if (!as_axis_symmetric(q, p)) {
    printf("FAIL %s: not recognised\n", name);
    ++fails;
    return;
  }
  double lo, hi;
  axis_sym_slice_bounds(b, p, 1 << 18, lo, hi);
  bool ok = lo <= exact + 1e-9 && exact <= hi + 1e-9;
  printf("  %-34s [%.9f, %.9f] exact %.9f  %s\n", name, lo, hi, exact,
    ok ? "brackets" : "*** VIOLATION ***");
  CK(ok, "%s", name);
}
static int run_all()
{

  // (1) On a rectangle, circle_poly_area must reproduce circle_rect_area, which
  //     is independently verified. Strongest available cross-check.
  double worst = 0;
  for (int t = 0; t < 20000; ++t) {
    double cx = U(-3, 3), cy = U(-3, 3), r = U(0.05, 3);
    double x0 = U(-4, 0), x1 = x0 + U(0.01, 5), y0 = U(-4, 0),
           y1 = y0 + U(0.01, 5);
    vector<double> px {x0, x1, x1, x0}, py {y0, y0, y1, y1};
    double a = circle_poly_area(cx, cy, r, px, py);
    double b = circle_rect_area(cx, cy, r, x0, x1, y0, y1);
    worst = std::max(worst, std::abs(a - b));
  }
  printf(
    "circle_poly vs circle_rect on rectangles: worst abs diff %.3e\n", worst);
  CK(worst < 1e-12, "poly/rect mismatch");

  // (2) Exact limits.
  vector<double> big {-9, 9, 9, -9}, bigy {-9, -9, 9, 9};
  CK(std::abs(circle_poly_area(0, 0, 1.5, big, bigy) - M_PI * 2.25) < 1e-12,
    "full disk");
  vector<double> tri {0, 4, 0}, triy {0, 0, 4};
  // disk of radius 100 covering the triangle -> triangle area 8
  CK(std::abs(circle_poly_area(0, 0, 100, tri, triy) - 8.0) < 1e-9,
    "polygon inside disk");
  // regular hexagon, apothem 1, circle radius 1 inscribed -> circle area
  vector<double> hx, hy;
  for (int k = 0; k < 6; ++k) {
    double th = M_PI / 6.0 + k * M_PI / 3.0, R = 2.0 / std::sqrt(3.0);
    hx.push_back(R * std::cos(th));
    hy.push_back(R * std::sin(th));
  }
  double hexa = poly_area(hx, hy);
  printf("hexagon area %.12f vs %.12f\n", std::abs(hexa), 2 * std::sqrt(3.0));
  CK(std::abs(std::abs(hexa) - 2 * std::sqrt(3.0)) < 1e-12, "hex area");
  double ins = circle_poly_area(0, 0, 1.0, hx, hy);
  printf("inscribed circle in hexagon: %.12f vs %.12f\n", ins, M_PI);
  CK(std::abs(ins - M_PI) < 1e-12, "inscribed circle");

  // (3) Additivity: splitting the polygon splits the area.
  worst = 0;
  for (int t = 0; t < 5000; ++t) {
    double cx = U(-2, 2), cy = U(-2, 2), r = U(0.1, 3);
    // random convex polygon: clip a big square by random halfplanes
    vector<double> px {-5, 5, 5, -5}, py {-5, -5, 5, 5};
    for (int k = 0; k < 3; ++k) {
      double th = U(0, 6.283);
      clip_poly(px, py, std::cos(th), std::sin(th), U(0.5, 3.0));
    }
    if (px.size() < 3)
      continue;
    double whole = circle_poly_area(cx, cy, r, px, py);
    double th = U(0, 6.283), ca = std::cos(th), sa = std::sin(th),
           c0 = U(-1, 1);
    vector<double> ax = px, ay = py, bx = px, by = py;
    clip_poly(ax, ay, ca, sa, c0);
    clip_poly(bx, by, -ca, -sa, -c0);
    double parts =
      circle_poly_area(cx, cy, r, ax, ay) + circle_poly_area(cx, cy, r, bx, by);
    worst = std::max(worst, std::abs(whole - parts));
  }
  printf("circle_poly additivity under polygon split: worst %.3e\n", worst);
  CK(worst < 1e-11, "poly additivity");

  // (4) Tangency sweep: a disk inscribed in a regular n-gon touches every wall.
  //     Sweep the radius through tangency to confirm nothing jumps.
  printf("tangency sweep (disk in hexagon, apothem 1):\n");
  for (double r : {0.999999, 0.9999999999, 1.0, 1.0000000001, 1.000001}) {
    double a = circle_poly_area(0, 0, r, hx, hy);
    double ref = (r <= 1.0) ? M_PI * r * r : -1;
    printf("   r=%.12f -> %.12f%s\n", r, a,
      ref > 0 ? (std::abs(a - ref) < 1e-9 ? "  (= pi r^2)" : "  *** JUMP ***")
              : "");
    if (ref > 0)
      CK(std::abs(a - ref) < 1e-9, "tangency at r=%.12f: %.12f vs %.12f", r, a,
        ref);
  }
  // Same for a triangle and a square, in case hexagon symmetry hides something.
  {
    vector<double> sx {-1, 1, 1, -1}, sy {-1, -1, 1, 1};
    CK(std::abs(circle_poly_area(0, 0, 1.0, sx, sy) - M_PI) < 1e-11,
      "disk in square");
    double R = 2.0;
    vector<double> tx, ty;
    for (int k = 0; k < 3; ++k) {
      double th = M_PI / 2 + k * 2 * M_PI / 3;
      tx.push_back(R * std::cos(th));
      ty.push_back(R * std::sin(th));
    }
    CK(std::abs(circle_poly_area(0, 0, R / 2, tx, ty) - M_PI * R * R / 4) <
         1e-11,
      "disk in triangle");
    printf("   disk inscribed in square and triangle: ok\n");
  }

  printf("diagonal quadrics beyond surfaces of revolution\n");
  // sphere r=2 (regression, sx=sy=1)
  {
    QuadricForm q;
    q.A[0] = q.A[1] = q.A[2] = 1;
    q.c = -4;
    check("sphere r=2", q, OctBox::from_aabb({-3, -3, -3}, {3, 3, 3}),
      4.0 / 3 * M_PI * 8);
  }
  // ellipsoid x^2/1 + y^2/4 + z^2/9 <= 1  -> a=1,b=2,c=3, V=4/3 pi abc
  {
    QuadricForm q;
    q.A[0] = 1.0;
    q.A[1] = 0.25;
    q.A[2] = 1.0 / 9.0;
    q.c = -1;
    check("ellipsoid a=1 b=2 c=3", q,
      OctBox::from_aabb({-4, -4, -4}, {4, 4, 4}), 4.0 / 3 * M_PI * 1 * 2 * 3);
  }
  // off-centre ellipsoid
  {
    QuadricForm q;
    q.A[0] = 1.0;
    q.A[1] = 0.25;
    q.A[2] = 1.0 / 9.0;
    double x0 = 0.4, y0 = -0.7, z0 = 1.1;
    q.b = {-2 * q.A[0] * x0, -2 * q.A[1] * y0, -2 * q.A[2] * z0};
    q.c = q.A[0] * x0 * x0 + q.A[1] * y0 * y0 + q.A[2] * z0 * z0 - 1;
    check("ellipsoid off-centre", q, OctBox::from_aabb({-5, -5, -5}, {5, 5, 5}),
      4.0 / 3 * M_PI * 1 * 2 * 3);
  }
  // elliptic cylinder along z: x^2/1 + y^2/(2.5^2) <= 1, height 4 -> pi*a*b*H
  {
    QuadricForm q;
    q.A[0] = 1.0;
    q.A[1] = 1.0 / 6.25;
    q.A[2] = 0.0;
    q.c = -1;
    check("elliptic cylinder a=1 b=2.5", q,
      OctBox::from_aabb({-4, -4, -2}, {4, 4, 2}), M_PI * 1 * 2.5 * 4);
  }
  // elliptic cone: x^2 + y^2/4 - z^2/9 <= 0 for z in [0,3]
  // cross-section at z: ellipse semi-axes (z/3, 2z/3) -> area pi*2z^2/9
  // volume = int_0^3 2 pi z^2/9 dz = 2 pi
  {
    QuadricForm q;
    q.A[0] = 1.0;
    q.A[1] = 0.25;
    q.A[2] = -1.0 / 9.0;
    check(
      "elliptic cone", q, OctBox::from_aabb({-2, -3, 0}, {2, 3, 3}), 2 * M_PI);
  }
  // circular cylinder via A=(4,4,0): 4x^2+4y^2-1<=0 -> r=0.5
  {
    QuadricForm q;
    q.A[0] = 4.0;
    q.A[1] = 4.0;
    q.c = -1.0;
    check("scaled circular cylinder r=0.5", q,
      OctBox::from_aabb({-1, -1, -1}, {1, 1, 1}), M_PI * 0.25 * 2);
  }
  // clipped: ellipsoid cut by the box
  {
    QuadricForm q;
    q.A[0] = 1.0;
    q.A[1] = 0.25;
    q.A[2] = 1.0 / 9.0;
    q.c = -1;
    AxisSymProfile p;
    as_axis_symmetric(q, p);
    OctBox b = OctBox::from_aabb({-1, -4, -4}, {0.5, 4, 4});
    double lo, hi;
    axis_sym_slice_bounds(b, p, 1 << 18, lo, hi);
    // reference: dense grid
    long in = 0;
    int M = 420;
    double vol = 1.5 * 8 * 8;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < M; ++k) {
          double x = -1 + 1.5 * (i + 0.5) / M, y = -4 + 8.0 * (j + 0.5) / M,
                 z = -4 + 8.0 * (k + 0.5) / M;
          if (x * x + 0.25 * y * y + z * z / 9.0 < 1.0)
            ++in;
        }
    double ref = vol * in / ((double)M * M * M);
    printf("  %-34s [%.9f, %.9f] grid ref %.6f  %s\n",
      "ellipsoid clipped by box", lo, hi, ref,
      (lo - 3e-3 <= ref && ref <= hi + 3e-3) ? "consistent"
                                             : "*** MISMATCH ***");
    CK(lo - 3e-3 <= ref && ref <= hi + 3e-3, "clipped ellipsoid");
  }
  printf(fails ? "\n%d FAILURES\n" : "\nall shape checks passed\n", fails);
  return fails ? 1 : 0;
}

TEST_CASE("volume_octree: cross-section kernels and diagonal quadrics",
  "[volume_octree]")
{
  // The body reports its own diagnostics on stdout and returns a failure count;
  // run with -s to see the tables even when everything passes.
  REQUIRE(run_all() == 0);
}
