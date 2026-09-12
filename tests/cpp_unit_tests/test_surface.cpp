#include <algorithm>
#include <cmath>
#include <memory>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <pugixml.hpp>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/surface.h"

using namespace openmc;

namespace {

template<typename T>
std::unique_ptr<T> make_surface(
  pugi::xml_document& doc, int id, const char* type, const char* coeffs)
{
  pugi::xml_node n = doc.append_child("surface");
  n.append_attribute("id") = id;
  n.append_attribute("type") = type;
  n.append_attribute("coeffs") = coeffs;
  return std::make_unique<T>(n);
}

} // anonymous namespace

TEST_CASE("Expand bounding box to include points")
{
  BoundingBox bbox = BoundingBox::inverted();

  bbox.expand({1.0, -2.0, 3.0});
  CHECK(bbox.min.x == 1.0);
  CHECK(bbox.min.y == -2.0);
  CHECK(bbox.min.z == 3.0);
  CHECK(bbox.max.x == 1.0);
  CHECK(bbox.max.y == -2.0);
  CHECK(bbox.max.z == 3.0);

  bbox.expand({-4.0, 0.0, 2.0});
  CHECK(bbox.min.x == -4.0);
  CHECK(bbox.min.y == -2.0);
  CHECK(bbox.min.z == 2.0);
  CHECK(bbox.max.x == 1.0);
  CHECK(bbox.max.y == 0.0);
  CHECK(bbox.max.z == 3.0);
}

TEST_CASE("General plane bounding box")
{
  pugi::xml_document doc;

  SECTION("Exactly axis-aligned planes bound one axis only")
  {
    // +x normal: the positive half-space starts at x = D/A
    auto px = make_surface<SurfacePlane>(doc, 1, "plane", "1.0 0.0 0.0 5.0");
    BoundingBox pos = px->bounding_box(true);
    CHECK(pos.min.x == Catch::Approx(5.0));
    CHECK(pos.min.y == -INFTY);
    CHECK(pos.min.z == -INFTY);
    CHECK(pos.max.x == INFTY);

    BoundingBox neg = px->bounding_box(false);
    CHECK(neg.max.x == Catch::Approx(5.0));
    CHECK(neg.min.x == -INFTY);
    CHECK(neg.max.y == INFTY);

    // -y normal: the sense of the bound flips with the sign of the normal
    auto ny = make_surface<SurfacePlane>(doc, 2, "plane", "0.0 -1.0 0.0 3.0");
    BoundingBox ny_pos = ny->bounding_box(true);
    CHECK(ny_pos.max.y == Catch::Approx(-3.0));
    CHECK(ny_pos.min.y == -INFTY);
    CHECK(ny_pos.min.x == -INFTY);

    BoundingBox ny_neg = ny->bounding_box(false);
    CHECK(ny_neg.min.y == Catch::Approx(-3.0));
    CHECK(ny_neg.max.y == INFTY);
  }

  SECTION("Coefficients need not be normalized")
  {
    // 4z - 10 = 0 is the same plane as z = 2.5
    auto pz = make_surface<SurfacePlane>(doc, 3, "plane", "0.0 0.0 4.0 10.0");
    CHECK(pz->bounding_box(true).min.z == Catch::Approx(2.5));
    CHECK(pz->bounding_box(false).max.z == Catch::Approx(2.5));
  }

  SECTION("Rotation roundoff is still axis aligned (issue #2632)")
  {
    // The surface s64 from the reported model: a y-plane at -1.3 written with
    // cos(pi/2) in the x and z slots.
    auto p = make_surface<SurfacePlane>(
      doc, 4, "plane", "6.123233995736766e-17 1.0 6.123233995736766e-17 -1.3");
    BoundingBox pos = p->bounding_box(true);
    CHECK(pos.min.y == Catch::Approx(-1.3));
    // The two off-axis directions must stay unbounded
    CHECK(pos.min.x == -INFTY);
    CHECK(pos.min.z == -INFTY);
    CHECK(pos.max.x == INFTY);
    CHECK(pos.max.z == INFTY);
  }

  SECTION("Oblique planes bound nothing")
  {
    auto p = make_surface<SurfacePlane>(
      doc, 5, "plane", "0.7071067811865476 0.0 0.7071067811865476 11.45");
    for (bool side : {false, true}) {
      BoundingBox bb = p->bounding_box(side);
      CHECK(bb.min.x == -INFTY);
      CHECK(bb.min.y == -INFTY);
      CHECK(bb.min.z == -INFTY);
      CHECK(bb.max.x == INFTY);
      CHECK(bb.max.y == INFTY);
      CHECK(bb.max.z == INFTY);
    }
  }

  SECTION("A plane tilted well beyond tolerance bounds nothing")
  {
    // 1e-5 is far outside PLANE_ALIGNMENT_TOL, so this must not be treated as
    // an x-plane, and in particular must not acquire a bound on y.
    auto p = make_surface<SurfacePlane>(doc, 6, "plane", "1.0 1e-5 0.0 5.0");
    BoundingBox bb = p->bounding_box(true);
    CHECK(bb.min.x == -INFTY);
    CHECK(bb.min.y == -INFTY);
  }

  SECTION("An off-axis coefficient above tolerance bounds nothing")
  {
    // Although the x component of the normalized normal is within
    // PLANE_ALIGNMENT_TOL of one, the y component is above the tolerance.
    auto p = make_surface<SurfacePlane>(doc, 7, "plane", "1.0 1e-11 0.0 5.0");
    for (bool side : {false, true}) {
      BoundingBox bb = p->bounding_box(side);
      CHECK(bb.min.x == -INFTY);
      CHECK(bb.min.y == -INFTY);
      CHECK(bb.min.z == -INFTY);
      CHECK(bb.max.x == INFTY);
      CHECK(bb.max.y == INFTY);
      CHECK(bb.max.z == INFTY);
    }
  }

  SECTION("A degenerate plane bounds nothing")
  {
    auto p = make_surface<SurfacePlane>(doc, 8, "plane", "0.0 0.0 0.0 1.0");
    BoundingBox bb = p->bounding_box(true);
    CHECK(bb.min.x == -INFTY);
    CHECK(bb.max.x == INFTY);
  }
}

TEST_CASE("Torus bounding box")
{
  pugi::xml_document doc;

  // x0 y0 z0 A B C, so the interior spans +/-B along the axis of revolution
  // and +/-(A + C) in the perpendicular directions.
  SECTION("x-torus")
  {
    auto t = make_surface<SurfaceXTorus>(
      doc, 1, "x-torus", "1.0 2.0 3.0 5.0 0.5 0.25");
    BoundingBox in = t->bounding_box(false);
    CHECK(in.min.x == Catch::Approx(0.5));
    CHECK(in.max.x == Catch::Approx(1.5));
    CHECK(in.min.y == Catch::Approx(-3.25));
    CHECK(in.max.y == Catch::Approx(7.25));
    CHECK(in.min.z == Catch::Approx(-2.25));
    CHECK(in.max.z == Catch::Approx(8.25));

    // The exterior of a torus is unbounded
    BoundingBox out = t->bounding_box(true);
    CHECK(out.min.x == -INFTY);
    CHECK(out.max.z == INFTY);
  }

  SECTION("y-torus")
  {
    auto t = make_surface<SurfaceYTorus>(
      doc, 2, "y-torus", "1.0 2.0 3.0 5.0 0.5 0.25");
    BoundingBox in = t->bounding_box(false);
    CHECK(in.min.y == Catch::Approx(1.5));
    CHECK(in.max.y == Catch::Approx(2.5));
    CHECK(in.min.x == Catch::Approx(-4.25));
    CHECK(in.max.x == Catch::Approx(6.25));
    CHECK(in.min.z == Catch::Approx(-2.25));
    CHECK(in.max.z == Catch::Approx(8.25));

    CHECK(t->bounding_box(true).max.y == INFTY);
  }

  SECTION("z-torus")
  {
    auto t = make_surface<SurfaceZTorus>(
      doc, 3, "z-torus", "1.0 2.0 3.0 5.0 0.5 0.25");
    BoundingBox in = t->bounding_box(false);
    CHECK(in.min.z == Catch::Approx(2.5));
    CHECK(in.max.z == Catch::Approx(3.5));
    CHECK(in.min.x == Catch::Approx(-4.25));
    CHECK(in.max.x == Catch::Approx(6.25));
    CHECK(in.min.y == Catch::Approx(-3.25));
    CHECK(in.max.y == Catch::Approx(7.25));

    CHECK(t->bounding_box(true).min.z == -INFTY);
  }
}

//==============================================================================
// Surface::distance_to_point
//
// Each closed form is checked against a brute-force search: sample the surface
// densely by ray-casting from far outside in many directions, and take the
// smallest distance found. The search can only ever overestimate, so the
// closed form is verified to be exactly the minimum rather than merely
// self-consistent.
//==============================================================================

namespace {

//! Smallest |q - r| over points q found on the surface by casting rays from a
//! sphere of radius `reach` around r in many directions.
double brute_force_distance(const Surface& surf, Position r, double reach)
{
  constexpr int N = 40000;
  double best = INFTY;
  // Deterministic near-uniform directions via the golden-angle spiral
  const double golden = PI * (3.0 - std::sqrt(5.0));
  for (int i = 0; i < N; ++i) {
    double z = 1.0 - 2.0 * (i + 0.5) / N;
    double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
    double phi = golden * i;
    Direction u {rho * std::cos(phi), rho * std::sin(phi), z};

    // March along the ray, taking every crossing it finds
    Position p = r;
    double travelled = 0.0;
    for (int hop = 0; hop < 8; ++hop) {
      double d = surf.distance(p, u, false);
      if (d == INFTY || travelled + d > reach)
        break;
      travelled += d;
      p = r + travelled * u;
      best = std::min(best, travelled);
      // step past the crossing just found
      travelled += 1e-9;
      p = r + travelled * u;
    }
  }
  return best;
}

void check_surface(const Surface& surf, Position r, double reach)
{
  double exact = surf.distance_to_point(r);
  REQUIRE(exact >= 0.0);
  double brute = brute_force_distance(surf, r, reach);
  REQUIRE(brute < INFTY);
  // The ray search samples directions, so it can only overshoot
  REQUIRE(exact <= brute + 1e-9);
  REQUIRE(exact == Catch::Approx(brute).epsilon(2e-3));
}

} // anonymous namespace

TEST_CASE("Exact distance from a point to a surface")
{
  pugi::xml_document doc;

  SECTION("planes")
  {
    auto xp = make_surface<SurfaceXPlane>(doc, 1, "x-plane", "3.0");
    REQUIRE(xp->distance_to_point({-1.0, 5.0, 7.0}) == Catch::Approx(4.0));
    REQUIRE(xp->distance_to_point({3.0, 0.0, 0.0}) == Catch::Approx(0.0));
    check_surface(*xp, {-1.0, 5.0, 7.0}, 20.0);

    auto yp = make_surface<SurfaceYPlane>(doc, 2, "y-plane", "-2.0");
    check_surface(*yp, {1.0, 4.0, -3.0}, 20.0);

    auto zp = make_surface<SurfaceZPlane>(doc, 3, "z-plane", "0.5");
    check_surface(*zp, {1.0, 4.0, -3.0}, 20.0);

    // 3x + 4y + 0z = 10, so the normal has length 5
    auto pl = make_surface<SurfacePlane>(doc, 4, "plane", "3.0 4.0 0.0 10.0");
    REQUIRE(pl->distance_to_point({0.0, 0.0, 0.0}) == Catch::Approx(2.0));
    check_surface(*pl, {0.0, 0.0, 0.0}, 20.0);
    check_surface(*pl, {-4.0, 3.0, 6.0}, 20.0);
  }

  SECTION("cylinders")
  {
    auto xc =
      make_surface<SurfaceXCylinder>(doc, 5, "x-cylinder", "0.0 0.0 2.0");
    REQUIRE(xc->distance_to_point({9.0, 3.0, 4.0}) == Catch::Approx(3.0));
    check_surface(*xc, {9.0, 3.0, 4.0}, 20.0);
    // inside the cylinder
    REQUIRE(xc->distance_to_point({1.0, 0.5, 0.0}) == Catch::Approx(1.5));
    check_surface(*xc, {1.0, 0.5, 0.0}, 20.0);

    auto yc =
      make_surface<SurfaceYCylinder>(doc, 6, "y-cylinder", "1.0 -1.0 3.0");
    check_surface(*yc, {5.0, 2.0, 4.0}, 30.0);

    auto zc =
      make_surface<SurfaceZCylinder>(doc, 7, "z-cylinder", "0.0 0.0 1.5");
    check_surface(*zc, {0.4, -0.3, 8.0}, 20.0);
  }

  SECTION("sphere")
  {
    auto sp = make_surface<SurfaceSphere>(doc, 8, "sphere", "1.0 2.0 3.0 4.0");
    REQUIRE(sp->distance_to_point({1.0, 2.0, 3.0}) == Catch::Approx(4.0));
    REQUIRE(sp->distance_to_point({1.0, 2.0, 12.0}) == Catch::Approx(5.0));
    check_surface(*sp, {1.0, 2.0, 12.0}, 30.0);
    check_surface(*sp, {2.0, 3.0, 4.0}, 30.0);
  }

  SECTION("cones")
  {
    // 45 degree cone about the z axis through the origin
    auto zk = make_surface<SurfaceZCone>(doc, 9, "z-cone", "0.0 0.0 0.0 1.0");
    // a point on the axis one unit from the apex is 1/sqrt(2) from the cone
    REQUIRE(zk->distance_to_point({0.0, 0.0, 1.0}) ==
            Catch::Approx(1.0 / std::sqrt(2.0)));
    REQUIRE(zk->distance_to_point({0.0, 0.0, 0.0}) == Catch::Approx(0.0));
    check_surface(*zk, {0.0, 0.0, 1.0}, 20.0);
    check_surface(*zk, {3.0, 0.0, 1.0}, 20.0);
    check_surface(*zk, {0.5, 0.5, -4.0}, 20.0);

    auto xk = make_surface<SurfaceXCone>(doc, 10, "x-cone", "1.0 0.0 0.0 0.25");
    check_surface(*xk, {4.0, 2.0, 1.0}, 30.0);

    auto yk = make_surface<SurfaceYCone>(doc, 11, "y-cone", "0.0 1.0 0.0 4.0");
    check_surface(*yk, {2.0, 3.0, -1.0}, 30.0);
  }

  SECTION("types without a closed form report it")
  {
    auto q = make_surface<SurfaceQuadric>(
      doc, 12, "quadric", "1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 -4.0");
    REQUIRE(q->distance_to_point({1.0, 1.0, 1.0}) < 0.0);
  }
}
