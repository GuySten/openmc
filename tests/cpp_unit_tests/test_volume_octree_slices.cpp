#include "openmc/volume_octree_math.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
using namespace openmc;
static int run_all()
{
  int fails = 0;
  // Sphere r=3 fully inside an 8 cm cube: bounds must bracket 4/3 pi r^3.
  QuadricForm q;
  q.A[0] = q.A[1] = q.A[2] = 1.0;
  q.c = -9.0;
  AxisSymProfile p;
  if (!as_axis_symmetric(q, p)) {
    printf("FAIL recognise sphere\n");
    return 1;
  }
  OctBox b = OctBox::from_aabb({-4, -4, -4}, {4, 4, 4});
  double exact = 4.0 / 3.0 * M_PI * 27;
  printf("sphere r=3, exact %.10f\n", exact);
  for (int N : {8, 32, 128, 512, 2048}) {
    double lo, hi;
    axis_sym_slice_bounds(b, p, N, lo, hi);
    printf("  N=%5d  [%.10f, %.10f]  spread %.3e  %s\n", N, lo, hi, hi - lo,
      (lo <= exact && exact <= hi) ? "brackets" : "*** VIOLATION ***");
    if (!(lo <= exact && exact <= hi))
      ++fails;
  }
  // Off-centre, clipped by the box on two sides.
  QuadricForm q2;
  q2.A[0] = q2.A[1] = q2.A[2] = 1.0;
  q2.b = {-2 * 1.0, -2 * 0.5, -2 * (-0.3)};
  q2.c = 1.0 + 0.25 + 0.09 - 4.0; // centre (1,.5,-.3), r=2
  AxisSymProfile p2;
  as_axis_symmetric(q2, p2);
  OctBox b2 = OctBox::from_aabb({-1, -1, -1}, {2, 2, 2});
  double lo, hi;
  axis_sym_slice_bounds(b2, p2, 4096, lo, hi);
  // reference: dense 3-D grid
  long inside = 0;
  int M = 400;
  double acc = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j)
      for (int k = 0; k < M; ++k) {
        double x = -1 + 3.0 * (i + 0.5) / M, y = -1 + 3.0 * (j + 0.5) / M,
               z = -1 + 3.0 * (k + 0.5) / M;
        double d =
          (x - 1) * (x - 1) + (y - 0.5) * (y - 0.5) + (z + 0.3) * (z + 0.3);
        if (d < 4.0)
          ++inside;
      }
  acc = 27.0 * inside / ((double)M * M * M);
  printf("clipped sphere: [%.8f, %.8f] spread %.2e   grid ref %.8f  %s\n", lo,
    hi, hi - lo, acc,
    (lo - 1e-3 <= acc && acc <= hi + 1e-3) ? "consistent" : "*** MISMATCH ***");
  if (!(lo - 1e-3 <= acc && acc <= hi + 1e-3))
    ++fails;
  // Cone
  QuadricForm q3;
  q3.A[0] = 1;
  q3.A[1] = 1;
  q3.A[2] = -0.25; // z-cone, slope^2=0.25
  AxisSymProfile p3;
  if (!as_axis_symmetric(q3, p3)) {
    printf("FAIL recognise cone\n");
    ++fails;
  }
  OctBox b3 = OctBox::from_aabb({-3, -3, 0}, {3, 3, 4});
  axis_sym_slice_bounds(b3, p3, 4096, lo, hi);
  double cone =
    M_PI * 0.25 * 64.0 / 3.0; // integral of pi*0.25*z^2 dz from 0..4
  printf("cone: [%.8f, %.8f] exact %.8f  %s\n", lo, hi, cone,
    (lo <= cone && cone <= hi) ? "brackets" : "*** VIOLATION ***");
  if (!(lo <= cone && cone <= hi))
    ++fails;
  printf(fails ? "\n%d FAILURES\n" : "\nslice integrator ok\n", fails);
  return fails ? 1 : 0;
}

TEST_CASE(
  "volume_octree: Riemann bounds for revolved surfaces", "[volume_octree]")
{
  // The body reports its own diagnostics on stdout and returns a failure count;
  // run with -s to see the tables even when everything passes.
  REQUIRE(run_all() == 0);
}
