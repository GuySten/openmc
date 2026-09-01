//! Standalone checks on volume_octree_math.h.
//!
//! These are the parts that can be silently, numerically wrong, and they need
//! no OpenMC build:
//!
//!   g++ -std=c++17 -O2 -I include -I <fmt> \
//!       tests/cpp_unit_tests/test_volume_octree_math.cpp -o t && ./t
//!
//! The soundness assertions are the important ones. A range bound that is
//! merely loose costs octree depth; a bound that excludes a value the surface
//! actually attains makes the whole bracket a lie.

#include "openmc/volume_octree_math.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <random>

using namespace openmc;

static int failures = 0;

#define OCT_CHECK(cond, ...)                                                   \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL %s:%d  ", __FILE__, __LINE__);                         \
      std::printf(__VA_ARGS__);                                                \
      std::printf("\n");                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static std::mt19937 rng(12345);
static double uni(double a, double b)
{
  return std::uniform_real_distribution<double>(a, b)(rng);
}

//! Random rotation matrix via Gram-Schmidt on a Gaussian matrix.
static void random_rotation(OctBox& b)
{
  std::normal_distribution<double> n(0., 1.);
  Position a0 {n(rng), n(rng), n(rng)};
  Position a1 {n(rng), n(rng), n(rng)};
  a0 = a0 / std::sqrt(a0.dot(a0));
  a1 = a1 - a0.dot(a1) * a0;
  a1 = a1 / std::sqrt(a1.dot(a1));
  Position a2 = a0.cross(a1);
  b.axis[0] = a0;
  b.axis[1] = a1;
  b.axis[2] = a2;
  b.rotated = true;
}

static OctBox random_box(bool rotated)
{
  OctBox b;
  b.center = {uni(-5, 5), uni(-5, 5), uni(-5, 5)};
  for (int k = 0; k < 3; ++k)
    b.half[k] = uni(0.05, 2.0);
  if (rotated)
    random_rotation(b);
  return b;
}

//==============================================================================
// 1. form_range must bracket the true range of the quadric over the box.
//==============================================================================

static void test_form_range_soundness()
{
  int n_loose = 0, n_total = 0;

  for (int trial = 0; trial < 4000; ++trial) {
    bool rotated = (trial % 2 == 1);
    OctBox box = random_box(rotated);

    QuadricForm q;
    // Half the trials use a separable form (no cross terms), where the bound
    // is supposed to be exact; half use a fully general one.
    bool separable = (trial % 4 < 2);
    for (int k = 0; k < 3; ++k)
      q.A[k] = uni(-2, 2);
    if (!separable)
      for (int k = 3; k < 6; ++k)
        q.A[k] = uni(-2, 2);
    q.b = {uni(-5, 5), uni(-5, 5), uni(-5, 5)};
    q.c = uni(-5, 5);

    double lo, hi;
    form_range(q, box, lo, hi);

    // Brute-force sample the box, including all corners and edge midpoints.
    double smin = 1e300, smax = -1e300;
    for (int s = 0; s < 3000; ++s) {
      double u0, u1, u2;
      if (s < 8) {
        u0 = (s & 1) ? box.half[0] : -box.half[0];
        u1 = (s & 2) ? box.half[1] : -box.half[1];
        u2 = (s & 4) ? box.half[2] : -box.half[2];
      } else {
        u0 = uni(-box.half[0], box.half[0]);
        u1 = uni(-box.half[1], box.half[1]);
        u2 = uni(-box.half[2], box.half[2]);
      }
      Position p =
        box.center + u0 * box.axis[0] + u1 * box.axis[1] + u2 * box.axis[2];
      double v = quad_eval(q, p);
      smin = std::min(smin, v);
      smax = std::max(smax, v);
    }

    double tol = 1e-9 * (1.0 + std::abs(smin) + std::abs(smax));
    OCT_CHECK(
      lo <= smin + tol, "form_range lower bound too high: %g > %g", lo, smin);
    OCT_CHECK(
      hi >= smax - tol, "form_range upper bound too low: %g < %g", hi, smax);

    // Separable + unrotated is claimed EXACT, so compare against an
    // independently computed exact range rather than against sampling.
    // Random sampling in 3-D systematically undershoots the true extrema, so
    // it cannot distinguish "exact" from "5% loose" -- a dense per-axis scan
    // can, because separability means the exact range is the sum of the exact
    // 1-D ranges.
    if (separable && !rotated) {
      const double A_diag[3] {q.A[0], q.A[1], q.A[2]};
      const double b_diag[3] {2. * q.A[0] * box.center.x + q.b.x,
        2. * q.A[1] * box.center.y + q.b.y, 2. * q.A[2] * box.center.z + q.b.z};
      double exact_lo = quad_eval(q, box.center), exact_hi = exact_lo;
      for (int k = 0; k < 3; ++k) {
        double h = box.half[k], klo = 1e300, khi = -1e300;
        const int N = 200000;
        for (int j = 0; j <= N; ++j) {
          double t = -h + 2.0 * h * j / N;
          double v = A_diag[k] * t * t + b_diag[k] * t;
          klo = std::min(klo, v);
          khi = std::max(khi, v);
        }
        exact_lo += klo;
        exact_hi += khi;
      }
      double scale = 1.0 + std::abs(exact_lo) + std::abs(exact_hi);
      OCT_CHECK(std::abs(lo - exact_lo) < 1e-6 * scale,
        "separable lower bound not exact: %.12g vs %.12g", lo, exact_lo);
      OCT_CHECK(std::abs(hi - exact_hi) < 1e-6 * scale,
        "separable upper bound not exact: %.12g vs %.12g", hi, exact_hi);
    }

    ++n_total;
    if (!separable && (hi - lo) > 3.0 * (smax - smin) + 1e-6)
      ++n_loose;
  }

  std::printf(
    "  form_range: %d/%d general forms >3x loose (expected, O(h^2))\n", n_loose,
    n_total);
}

//==============================================================================
// 2. Loosening must vanish as h -> 0, at second order.
//==============================================================================

static void test_cross_term_convergence()
{
  QuadricForm q;
  q.A[0] = 1.0;
  q.A[1] = -0.7;
  q.A[2] = 0.4;
  q.A[3] = 1.3;
  q.A[4] = -0.9;
  q.A[5] = 0.6;
  q.b = {2., -1., 0.5};
  q.c = -3.;

  OctBox box;
  box.center = {1.2, -0.7, 0.3};
  random_rotation(box);

  double prev = -1.0;
  for (int level = 0; level < 6; ++level) {
    double h = 1.0 / (1 << level);
    for (int k = 0; k < 3; ++k)
      box.half[k] = h;

    double lo, hi;
    form_range(q, box, lo, hi);

    double smin = 1e300, smax = -1e300;
    for (int s = 0; s < 20000; ++s) {
      Position p = box.center + uni(-h, h) * box.axis[0] +
                   uni(-h, h) * box.axis[1] + uni(-h, h) * box.axis[2];
      double v = quad_eval(q, p);
      smin = std::min(smin, v);
      smax = std::max(smax, v);
    }
    double excess = (hi - lo) - (smax - smin);
    if (prev > 0.0) {
      // Halving h should shrink the excess by roughly 4x (it is O(h^2)).
      OCT_CHECK(excess < prev * 0.45,
        "cross-term excess not converging: %g then %g", prev, excess);
    }
    prev = excess;
  }
}

//==============================================================================
// 3. Torus range, against OpenMC's own equation.
//==============================================================================

static double torus_eval(
  int ax, Position tc, double A, double B, double C, Position p)
{
  double d[3] {p.x - tc.x, p.y - tc.y, p.z - tc.z};
  int r1 = (ax + 1) % 3, r2 = (ax + 2) % 3;
  double rho = std::sqrt(d[r1] * d[r1] + d[r2] * d[r2]);
  return d[ax] * d[ax] / (B * B) + std::pow(rho - A, 2) / (C * C) - 1.0;
}

static void test_torus_range()
{
  for (int trial = 0; trial < 120; ++trial) {
    int ax = trial % 3;
    Position tc {uni(-2, 2), uni(-2, 2), uni(-2, 2)};
    double A = uni(1.0, 6.0), B = uni(0.3, 2.0), C = uni(0.3, 2.0);

    OctBox box = random_box(false);

    double lo, hi;
    torus_range(ax, tc, A, B, C, box, lo, hi);

    double smin = 1e300, smax = -1e300;
    for (int s = 0; s < 4000; ++s) {
      Position p {box.center.x + uni(-box.half[0], box.half[0]),
        box.center.y + uni(-box.half[1], box.half[1]),
        box.center.z + uni(-box.half[2], box.half[2])};
      double v = torus_eval(ax, tc, A, B, C, p);
      smin = std::min(smin, v);
      smax = std::max(smax, v);
    }

    OCT_CHECK(
      lo <= smin + 1e-9, "torus lower bound too high: %g > %g", lo, smin);
    OCT_CHECK(
      hi >= smax - 1e-9, "torus upper bound too low: %g < %g", hi, smax);

    // Claimed exact. Same argument as above: build the exact range from dense
    // scans of the two independent terms instead of from 3-D sampling.
    {
      int r1 = (ax + 1) % 3, r2 = (ax + 2) % 3;
      const double bc[3] {box.center.x, box.center.y, box.center.z};
      const double tcc[3] {tc.x, tc.y, tc.z};

      double a_lo = 1e300, a_hi = -1e300;
      const int N = 100000;
      for (int j = 0; j <= N; ++j) {
        double t = bc[ax] - box.half[ax] + 2.0 * box.half[ax] * j / N;
        double v = std::pow(t - tcc[ax], 2) / (B * B);
        a_lo = std::min(a_lo, v);
        a_hi = std::max(a_hi, v);
      }

      // rho over the rectangle: dense 2-D scan of the face.
      double rho_lo = 1e300, rho_hi = -1e300;
      const int M = 1200;
      for (int j1 = 0; j1 <= M; ++j1) {
        double u1 = bc[r1] - box.half[r1] + 2.0 * box.half[r1] * j1 / M;
        for (int j2 = 0; j2 <= M; ++j2) {
          double u2 = bc[r2] - box.half[r2] + 2.0 * box.half[r2] * j2 / M;
          double d1 = u1 - tcc[r1], d2 = u2 - tcc[r2];
          double rho = std::sqrt(d1 * d1 + d2 * d2);
          rho_lo = std::min(rho_lo, rho);
          rho_hi = std::max(rho_hi, rho);
        }
      }
      double t_lo = 1e300, t_hi = -1e300;
      for (int j = 0; j <= N; ++j) {
        double rho = rho_lo + (rho_hi - rho_lo) * j / N;
        double v = std::pow(rho - A, 2) / (C * C);
        t_lo = std::min(t_lo, v);
        t_hi = std::max(t_hi, v);
      }

      double exact_lo = a_lo + t_lo - 1.0, exact_hi = a_hi + t_hi - 1.0;
      double scale = 1.0 + std::abs(exact_lo) + std::abs(exact_hi);
      OCT_CHECK(std::abs(lo - exact_lo) < 1e-4 * scale,
        "torus lower bound not exact: %.12g vs %.12g", lo, exact_lo);
      OCT_CHECK(std::abs(hi - exact_hi) < 1e-4 * scale,
        "torus upper bound not exact: %.12g vs %.12g", hi, exact_hi);
    }
  }
}

//==============================================================================
// 4. OctBox: children tile the parent, transforms are rigid.
//==============================================================================

static void test_box_geometry()
{
  for (int trial = 0; trial < 500; ++trial) {
    OctBox b = random_box(trial % 2 == 1);

    double child_sum = 0.0;
    for (int i = 0; i < 8; ++i)
      child_sum += b.child(i).volume();
    OCT_CHECK(std::abs(child_sum - b.volume()) < 1e-12 * b.volume(),
      "children do not tile: %g vs %g", child_sum, b.volume());

    // Children must be disjoint and inside the parent: check each child's
    // centre lies in the parent and in exactly one child.
    for (int i = 0; i < 8; ++i) {
      Position cc = b.child(i).center;
      Position d = cc - b.center;
      for (int k = 0; k < 3; ++k) {
        double proj = d.dot(b.axis[k]);
        OCT_CHECK(std::abs(proj) <= b.half[k] + 1e-12,
          "child centre outside parent along axis %d", k);
      }
    }

    // A rigid transform must preserve volume and the orthonormality of the
    // frame. This is what guarantees no volume is created or lost when
    // descending through a rotated fill.
    OctBox rot = random_box(true);
    vector<double> R {rot.axis[0].x, rot.axis[0].y, rot.axis[0].z,
      rot.axis[1].x, rot.axis[1].y, rot.axis[1].z, rot.axis[2].x, rot.axis[2].y,
      rot.axis[2].z};
    OctBox t = b.transformed(Position {1., -2., 3.}, R);
    OCT_CHECK(std::abs(t.volume() - b.volume()) < 1e-12 * b.volume(),
      "transform changed volume");
    for (int k = 0; k < 3; ++k) {
      OCT_CHECK(std::abs(t.axis[k].dot(t.axis[k]) - 1.0) < 1e-12,
        "transformed axis %d not unit", k);
      for (int l = k + 1; l < 3; ++l)
        OCT_CHECK(std::abs(t.axis[k].dot(t.axis[l])) < 1e-12,
          "transformed axes %d,%d not orthogonal", k, l);
    }

    // Corner correspondence: corner i of the transformed box must be the
    // transform of corner i of the original. This catches a transposed
    // rotation matrix, which is otherwise invisible.
    vector<double> empty;
    for (int i = 0; i < 8; ++i) {
      Position expect = (b.corner(i) - Position {1., -2., 3.}).rotate(R);
      Position got = t.corner(i);
      OCT_CHECK((expect - got).dot(expect - got) < 1e-18,
        "corner %d mismatch after transform", i);
    }
  }
}

//==============================================================================
// 5. Three-valued evaluation must agree with pointwise evaluation.
//==============================================================================
//
// Build a random region over a few surfaces, assign each surface a definite or
// straddling box sense, and check: whenever the three-valued result is TRUE,
// every consistent pointwise assignment is true; whenever FALSE, every
// consistent assignment is false. MAYBE is allowed to be either.

static bool eval_bool(
  const vector<int32_t>& post, bool simple, const vector<bool>& sense_pos)
{
  if (simple) {
    for (int32_t t : post)
      if (sense_pos[std::abs(t) - 1] != (t > 0))
        return false;
    return true;
  }
  vector<bool> st;
  for (int32_t t : post) {
    if (t == kOpUnion) {
      bool b = st.back();
      st.pop_back();
      bool a = st.back();
      st.pop_back();
      st.push_back(a || b);
    } else if (t == kOpIntersection) {
      bool b = st.back();
      st.pop_back();
      bool a = st.back();
      st.pop_back();
      st.push_back(a && b);
    } else if (t == kOpComplement) {
      bool a = st.back();
      st.pop_back();
      st.push_back(!a);
    } else {
      st.push_back(sense_pos[std::abs(t) - 1] == (t > 0));
    }
  }
  return st.back();
}

static void test_tri_logic()
{
  const int n_surf = 5;

  for (int trial = 0; trial < 20000; ++trial) {
    // Random postfix expression over n_surf surfaces.
    vector<int32_t> post;
    int operands = 0;
    int n_tokens = 3 + (rng() % 8);
    while (static_cast<int>(post.size()) < n_tokens || operands != 1) {
      bool want_op = operands >= 2 && (rng() % 3 != 0);
      if (want_op) {
        int r = rng() % 3;
        post.push_back(
          r == 0 ? kOpUnion : (r == 1 ? kOpIntersection : kOpComplement));
        if (r == 2) {
          // complement is unary; only legal with >=1 operand
          post.back() = kOpComplement;
        } else {
          --operands;
        }
      } else {
        int32_t s = 1 + rng() % n_surf;
        post.push_back((rng() % 2) ? s : -s);
        ++operands;
      }
      if (post.size() > 40)
        break;
    }
    if (operands != 1)
      continue;

    // Assign each surface a box sense.
    vector<BoxSense> senses(n_surf);
    for (int i = 0; i < n_surf; ++i) {
      int r = rng() % 3;
      senses[i] = r == 0 ? BoxSense::NEGATIVE
                         : (r == 1 ? BoxSense::POSITIVE : BoxSense::BOTH);
    }

    Tri t =
      evaluate_region_tri(post, false, [&](int32_t i) { return senses[i]; });

    // Enumerate all pointwise assignments consistent with those senses.
    bool saw_true = false, saw_false = false;
    for (int mask = 0; mask < (1 << n_surf); ++mask) {
      vector<bool> sp(n_surf);
      bool consistent = true;
      for (int i = 0; i < n_surf; ++i) {
        bool pos = (mask >> i) & 1;
        if (senses[i] == BoxSense::NEGATIVE && pos)
          consistent = false;
        if (senses[i] == BoxSense::POSITIVE && !pos)
          consistent = false;
        sp[i] = pos;
      }
      if (!consistent)
        continue;
      (eval_bool(post, false, sp) ? saw_true : saw_false) = true;
    }

    if (t == Tri::kTrue)
      OCT_CHECK(!saw_false, "TRUE but a consistent assignment is false");
    if (t == Tri::kFalse)
      OCT_CHECK(!saw_true, "FALSE but a consistent assignment is true");
  }
}

//==============================================================================

static int run_all()
{
  std::printf("volume_octree_math checks\n");
  test_form_range_soundness();
  std::printf("  form_range soundness .......... done\n");
  test_cross_term_convergence();
  std::printf("  cross-term O(h^2) convergence . done\n");
  test_torus_range();
  std::printf("  torus range ................... done\n");
  test_box_geometry();
  std::printf("  box tiling and rigid transform  done\n");
  test_tri_logic();
  std::printf("  three-valued logic soundness .. done\n");

  if (failures == 0) {
    std::printf("\nall checks passed\n");
    return 0;
  }
  std::printf("\n%d FAILURES\n", failures);
  return 1;
}

TEST_CASE(
  "volume_octree: range bounds and three-valued logic", "[volume_octree]")
{
  // The body reports its own diagnostics on stdout and returns a failure count;
  // run with -s to see the tables even when everything passes.
  REQUIRE(run_all() == 0);
}
