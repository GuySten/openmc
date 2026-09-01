#include "openmc/volume_octree_math.h"
#include <cstdio>
#include <random>
using namespace openmc;
static int fails = 0;
static std::mt19937 rng(7);
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

int main()
{
  // (1) Additivity. Exact identity, independent of any reference: splitting the
  // rectangle must split the area. Catches case-analysis bugs precisely.
  double worst_add = 0;
  for (int t = 0; t < 20000; ++t) {
    double cx = U(-3, 3), cy = U(-3, 3), r = U(0.05, 3);
    double x0 = U(-4, 0), x1 = x0 + U(0.01, 5), y0 = U(-4, 0),
           y1 = y0 + U(0.01, 5);
    double xm = U(x0, x1), ym = U(y0, y1);
    double whole = circle_rect_area(cx, cy, r, x0, x1, y0, y1);
    double parts = circle_rect_area(cx, cy, r, x0, xm, y0, ym) +
                   circle_rect_area(cx, cy, r, xm, x1, y0, ym) +
                   circle_rect_area(cx, cy, r, x0, xm, ym, y1) +
                   circle_rect_area(cx, cy, r, xm, x1, ym, y1);
    worst_add = std::max(worst_add, std::abs(whole - parts));
  }
  printf(
    "circle_rect additivity: worst absolute discrepancy %.3e\n", worst_add);
  CK(worst_add < 1e-12, "additivity broken");

  // (2) Containment limits, exact.
  CK(std::abs(circle_rect_area(0, 0, 1.5, -9, 9, -9, 9) - M_PI * 2.25) < 1e-13,
    "full disk");
  CK(std::abs(circle_rect_area(0, 0, 100, -1, 2, -3, 4) - (3.0 * 7.0)) < 1e-10,
    "rect inside disk");
  CK(circle_rect_area(0, 0, 1, 5, 6, 5, 6) == 0.0, "disjoint");

  // (3) Converged quadrature, exact in y and Richardson-checked in x.
  double worst_q = 0;
  for (int t = 0; t < 300; ++t) {
    double cx = U(-2, 2), cy = U(-2, 2), r = U(0.2, 3);
    double x0 = U(-3, 0), x1 = x0 + U(0.5, 4), y0 = U(-3, 0),
           y1 = y0 + U(0.5, 4);
    double a = circle_rect_area(cx, cy, r, x0, x1, y0, y1);
    auto quad = [&](int N) {
      double acc = 0, dx = (x1 - x0) / N;
      for (int i = 0; i < N; ++i) {
        double x = x0 + (i + 0.5) * dx, dxc = x - cx, rr = r * r - dxc * dxc;
        if (rr <= 0)
          continue;
        double g = std::sqrt(rr);
        double lo = std::max(y0, cy - g), hi = std::min(y1, cy + g);
        if (hi > lo)
          acc += (hi - lo) * dx;
      }
      return acc;
    };
    double q1 = quad(200000), q2 = quad(400000);
    double refine = std::abs(q2 - q1); // quadrature's own error scale
    double err = std::abs(a - q2);
    worst_q = std::max(worst_q, err / (r * r));
    CK(err < 10 * refine + 1e-9,
      "circle_rect %g vs converged %g (err %.2e, quad refine %.2e)", a, q2, err,
      refine);
  }
  printf("circle_rect vs converged quadrature: worst err/r^2 %.3e\n", worst_q);

  // (4) box_halfspace_volume: additivity under box subdivision is again exact.
  std::normal_distribution<double> nd(0., 1.);
  double worst_hs = 0;
  int skipped = 0;
  for (int t = 0; t < 20000; ++t) {
    OctBox b;
    b.center = {U(-2, 2), U(-2, 2), U(-2, 2)};
    for (int k = 0; k < 3; ++k)
      b.half[k] = U(0.1, 2.0);
    if (t % 2) {
      Position a0 {nd(rng), nd(rng), nd(rng)}, a1 {nd(rng), nd(rng), nd(rng)};
      a0 = a0 / std::sqrt(a0.dot(a0));
      a1 = a1 - a0.dot(a1) * a0;
      a1 = a1 / std::sqrt(a1.dot(a1));
      b.axis[0] = a0;
      b.axis[1] = a1;
      b.axis[2] = a0.cross(a1);
      b.rotated = true;
    }
    Position n {nd(rng), nd(rng), nd(rng)};
    if (t % 7 == 0)
      n = {0, 0, 1};
    if (t % 11 == 0)
      n = {1, 1, 0};
    double d = U(-3, 3), v;
    if (!box_halfspace_volume(b, n, d, v)) {
      ++skipped;
      continue;
    }
    double parts = 0;
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
      double vi;
      if (!box_halfspace_volume(b.child(i), n, d, vi)) {
        ok = false;
        break;
      }
      parts += vi;
    }
    if (ok)
      worst_hs = std::max(worst_hs, std::abs(v - parts) / b.volume());
  }
  // Not machine epsilon: the subset formula subtracts terms of similar
  // magnitude, so ~1e-11 relative is inherent cancellation rather than a bug.
  // INTEGRATOR_GUARD exists to keep the bracket honest despite exactly this.
  printf("box_halfspace additivity: worst err/box %.3e (%d ill-conditioned, "
         "refused)\n",
    worst_hs, skipped);
  CK(worst_hs < 1e-9, "halfspace additivity worse than cancellation explains");
  CK(worst_hs < INTEGRATOR_GUARD, "cancellation exceeds INTEGRATOR_GUARD");

  // (5) box_halfspace_volume against MC, which only needs to catch gross
  // errors.
  double worst_mc = 0;
  for (int t = 0; t < 400; ++t) {
    OctBox b;
    b.center = {U(-2, 2), U(-2, 2), U(-2, 2)};
    for (int k = 0; k < 3; ++k)
      b.half[k] = U(0.1, 2.0);
    if (t % 2) {
      Position a0 {nd(rng), nd(rng), nd(rng)}, a1 {nd(rng), nd(rng), nd(rng)};
      a0 = a0 / std::sqrt(a0.dot(a0));
      a1 = a1 - a0.dot(a1) * a0;
      a1 = a1 / std::sqrt(a1.dot(a1));
      b.axis[0] = a0;
      b.axis[1] = a1;
      b.axis[2] = a0.cross(a1);
      b.rotated = true;
    }
    Position n {nd(rng), nd(rng), nd(rng)};
    double d = U(-3, 3), v;
    if (!box_halfspace_volume(b, n, d, v))
      continue;
    int N = 200000;
    long hit = 0;
    for (int i = 0; i < N; ++i) {
      Position p = b.center + U(-b.half[0], b.half[0]) * b.axis[0] +
                   U(-b.half[1], b.half[1]) * b.axis[1] +
                   U(-b.half[2], b.half[2]) * b.axis[2];
      if (n.dot(p) <= d)
        ++hit;
    }
    worst_mc =
      std::max(worst_mc, std::abs(v - b.volume() * hit / N) / b.volume());
  }
  printf("box_halfspace vs MC: worst err/box %.3e (MC 1-sigma ~ %.1e)\n",
    worst_mc, 0.5 / std::sqrt(200000.0));
  CK(worst_mc < 6e-3, "halfspace disagrees with MC");

  // exact half-box
  OctBox b = OctBox::from_aabb({-1, -2, -3}, {1, 2, 3});
  double v;
  box_halfspace_volume(b, {0, 0, 1}, 0.0, v);
  CK(std::abs(v - b.volume() / 2) < 1e-12, "half box %.15f", v);

  // === coincident halfspaces must not double-count their shared face ===
  // A volume calculation's bounding box routinely sits exactly on the model's
  // own bounding planes, and each coincident pair contributes a full face to
  // the divergence sum unless they are deduplicated. This produced a 4/3 error
  // on a hex prism.
  {
    OctBox cb = OctBox::from_aabb({-2, -2, -1.5}, {2, 2, 1.5});
    // z planes exactly on the box faces, plus six hex sides
    vector<Position> hn;
    vector<double> hd;
    for (int k = 0; k < 6; ++k) {
      double th = k * M_PI / 3.0;
      hn.push_back({std::cos(th), std::sin(th), 0});
      hd.push_back(1.0);
    }
    hn.push_back({0, 0, 1});
    hd.push_back(1.5);
    hn.push_back({0, 0, -1});
    hd.push_back(1.5);
    double v = box_planes_volume(cb, hn, hd);
    double ex = 2.0 * std::sqrt(3.0) * 1.0 * 3.0;
    printf(
      "hex prism with planes coincident on box faces: %.12f vs %.12f\n", v, ex);
    CK(std::abs(v - ex) < 1e-11,
      "coincident planes double-counted (%.6f vs %.6f)", v, ex);

    double sum = 0;
    for (int mask = 0; mask < (1 << 8); ++mask) {
      vector<Position> nn;
      vector<double> dd;
      for (int i = 0; i < 8; ++i) {
        if ((mask >> i) & 1) {
          nn.push_back(-1.0 * hn[i]);
          dd.push_back(-hd[i]);
        } else {
          nn.push_back(hn[i]);
          dd.push_back(hd[i]);
        }
      }
      sum += box_planes_volume(cb, nn, dd);
    }
    printf("  partition sum %.12f vs box %.12f\n", sum, cb.volume());
    CK(std::abs(sum - cb.volume()) < 1e-9 * cb.volume(),
      "coincident-plane partition");

    // duplicate a plane many times: the answer must not move
    for (int r = 0; r < 3; ++r) {
      hn.push_back({0, 0, 1});
      hd.push_back(1.5);
    }
    double v2 = box_planes_volume(cb, hn, hd);
    CK(std::abs(v2 - ex) < 1e-11, "repeated duplicates (%.6f)", v2);
    printf("  with 3 extra duplicate planes: %.12f\n", v2);
  }

  printf(fails ? "\n%d FAILURES\n" : "\nall kernel checks passed\n", fails);
  return fails ? 1 : 0;
}
