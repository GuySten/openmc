#include "openmc/volume_octree_math.h"
#include <cstdio>
#include <random>
using namespace openmc;
static int fails = 0;
static std::mt19937 rng(31);
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
static std::normal_distribution<double> nd(0., 1.);
static void rot(OctBox& b)
{
  Position a0 {nd(rng), nd(rng), nd(rng)}, a1 {nd(rng), nd(rng), nd(rng)};
  a0 = a0 / std::sqrt(a0.dot(a0));
  a1 = a1 - a0.dot(a1) * a0;
  a1 = a1 / std::sqrt(a1.dot(a1));
  b.axis[0] = a0;
  b.axis[1] = a1;
  b.axis[2] = a0.cross(a1);
  b.rotated = true;
}

int main()
{
  // === box_planes_volume: agrees with the single-plane closed form ===
  double worst = 0;
  for (int t = 0; t < 4000; ++t) {
    OctBox b;
    b.center = {U(-2, 2), U(-2, 2), U(-2, 2)};
    for (int k = 0; k < 3; ++k)
      b.half[k] = U(0.1, 2.0);
    if (t % 2)
      rot(b);
    Position n {nd(rng), nd(rng), nd(rng)};
    double d = U(-3, 3);
    double ref;
    if (!box_halfspace_volume(b, n, d, ref))
      continue;
    double got = box_planes_volume(b, {n}, {d});
    worst = std::max(worst, std::abs(got - ref) / b.volume());
  }
  printf("box_planes vs single-plane closed form: worst err/box %.3e\n", worst);
  CK(worst < 1e-9, "polytope disagrees with closed form");

  // === partition identity: the 2^m sign combinations must tile the box ===
  worst = 0;
  for (int t = 0; t < 3000; ++t) {
    OctBox b;
    b.center = {U(-1, 1), U(-1, 1), U(-1, 1)};
    for (int k = 0; k < 3; ++k)
      b.half[k] = U(0.2, 1.5);
    if (t % 2)
      rot(b);
    int m = 1 + rng() % 4;
    vector<Position> n;
    vector<double> d;
    for (int i = 0; i < m; ++i) {
      n.push_back({nd(rng), nd(rng), nd(rng)});
      // aim the plane near the box so it actually cuts
      d.push_back(n.back().dot(b.center) + U(-1.0, 1.0));
    }
    double sum = 0;
    for (int mask = 0; mask < (1 << m); ++mask) {
      vector<Position> nn;
      vector<double> dd;
      for (int i = 0; i < m; ++i) {
        if ((mask >> i) & 1) {
          nn.push_back(-1.0 * n[i]);
          dd.push_back(-d[i]);
        } else {
          nn.push_back(n[i]);
          dd.push_back(d[i]);
        }
      }
      sum += box_planes_volume(b, nn, dd);
    }
    worst = std::max(worst, std::abs(sum - b.volume()) / b.volume());
  }
  printf(
    "box_planes partition identity (2^m pieces tile the box): worst %.3e\n",
    worst);
  CK(worst < 1e-9, "sign combinations do not tile");

  // === additivity under box subdivision ===
  worst = 0;
  for (int t = 0; t < 2000; ++t) {
    OctBox b;
    b.center = {U(-1, 1), U(-1, 1), U(-1, 1)};
    for (int k = 0; k < 3; ++k)
      b.half[k] = U(0.2, 1.5);
    if (t % 2)
      rot(b);
    int m = 1 + rng() % 3;
    vector<Position> n;
    vector<double> d;
    for (int i = 0; i < m; ++i) {
      n.push_back({nd(rng), nd(rng), nd(rng)});
      d.push_back(n.back().dot(b.center) + U(-1.0, 1.0));
    }
    double v = box_planes_volume(b, n, d), parts = 0;
    for (int i = 0; i < 8; ++i)
      parts += box_planes_volume(b.child(i), n, d);
    worst = std::max(worst, std::abs(v - parts) / b.volume());
  }
  printf("box_planes additivity: worst %.3e\n", worst);
  CK(worst < 1e-9, "polytope additivity broken");

  // === exact reference: unit cube cut by x+y+z <= 1 -> 1/6 ===
  OctBox c = OctBox::from_aabb({0, 0, 0}, {1, 1, 1});
  double v = box_planes_volume(c, {Position {1, 1, 1}}, {1.0});
  printf("cube cut by x+y+z<=1: %.15f vs %.15f\n", v, 1.0 / 6.0);
  CK(std::abs(v - 1.0 / 6.0) < 1e-12, "tetra");
  // two planes: x+y+z<=1 and x<=0.5
  v =
    box_planes_volume(c, {Position {1, 1, 1}, Position {1, 0, 0}}, {1.0, 0.5});
  double ref =
    1.0 / 6.0 - (1.0 / 6.0) * std::pow(0.5, 3); // similar tetra removed
  printf("plus x<=0.5: %.15f vs %.15f\n", v, ref);
  CK(std::abs(v - ref) < 1e-12, "two planes");

  // === torus slice bounds ===
  // Torus about z, centre origin, A=5 major, B=C=1 -> volume 2 pi^2 A B C
  AxisSymProfile tp = torus_profile(2, {0, 0, 0}, 5.0, 1.0, 1.5);
  OctBox tb = OctBox::from_aabb({-8, -8, -2}, {8, 8, 2});
  double exact = 2.0 * M_PI * M_PI * 5.0 * 1.0 * 1.5;
  printf("torus exact %.10f\n", exact);
  for (int N : {32, 256, 2048, 16384}) {
    double lo, hi;
    axis_sym_slice_bounds(tb, tp, N, lo, hi);
    printf("  N=%6d [%.8f, %.8f] spread %.3e %s\n", N, lo, hi, hi - lo,
      (lo <= exact && exact <= hi) ? "brackets" : "*** VIOLATION ***");
    CK(lo <= exact && exact <= hi, "torus bracket N=%d", N);
  }
  // self-intersecting torus (C > A) still monotone
  AxisSymProfile tp2 = torus_profile(2, {0.3, -0.2, 0.1}, 1.0, 0.8, 1.4);
  OctBox tb2 = OctBox::from_aabb({-4, -4, -2}, {4, 4, 2});
  double lo, hi;
  axis_sym_slice_bounds(tb2, tp2, 32768, lo, hi);
  // A midpoint grid converges only as O(h) here (the boundary is a surface), so
  // check that it is CONVERGING TOWARD the bracket rather than picking a
  // tolerance out of the air.
  double vol = 8.0 * 8.0 * 4.0, prev = 0;
  printf("self-intersecting torus bracket: [%.6f, %.6f]\n", lo, hi);
  for (int M : {200, 400, 800}) {
    long in = 0;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < M / 2; ++k) {
          double x = -4 + 8.0 * (i + 0.5) / M, y = -4 + 8.0 * (j + 0.5) / M,
                 z = -2 + 4.0 * (k + 0.5) / (M / 2);
          double rho = std::sqrt((x - 0.3) * (x - 0.3) + (y + 0.2) * (y + 0.2));
          double q = (z - 0.1) * (z - 0.1) / (0.8 * 0.8) +
                     (rho - 1.0) * (rho - 1.0) / (1.4 * 1.4) - 1.0;
          if (q < 0)
            ++in;
        }
    double ref2 = vol * in / ((double)M * M * (M / 2));
    printf("   grid M=%4d -> %.6f   gap to bracket %+.2e%s\n", M, ref2,
      ref2 < lo ? ref2 - lo : (ref2 > hi ? ref2 - hi : 0.0), prev ? "" : "");
    if (prev) {
      double g1 = std::abs(prev - 0.5 * (lo + hi)),
             g2 = std::abs(ref2 - 0.5 * (lo + hi));
      printf("        gap shrank %.2fx on halving h\n", g1 / g2);
      CK(g2 < g1, "grid not converging toward bracket");
    }
    prev = ref2;
  }

  printf(fails ? "\n%d FAILURES\n" : "\nall gap-closing kernels ok\n", fails);
  return fails ? 1 : 0;
}
