//! End-to-end checks on the base-case integrators, standalone.
//!
//! Mirrors OctreeVolumeCalculation::try_integrate against a minimal harness so
//! the integrator logic -- sign enumeration, disjointness reasoning, annulus
//! collapse, piece ownership -- is testable without an OpenMC build.
//!
//!   g++ -std=c++17 -O2 -I include -I <fmt> \
//!       tests/cpp_unit_tests/test_volume_octree_integrators.cpp \
//!       src/position.cpp -o t && ./t
//!
//! Every case asserts the bracket contains the analytic volume. That is the
//! property everything else rests on; tightness is reported but secondary.

#include "openmc/volume_octree_math.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

using namespace openmc;

static int failures = 0;
#define OCT_CHECK(cond, ...)                                                   \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("  FAIL ");                                                  \
      std::printf(__VA_ARGS__);                                                \
      std::printf("\n");                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

//==============================================================================
// Minimal mirror of the driver
//==============================================================================

struct Proxy {
  enum class Kind { QUADRIC, TORUS, OPAQUE } kind {Kind::QUADRIC};
  QuadricForm q;
  int ax {0};
  Position tc;
  double A {0}, B {0}, C {0};
};

static vector<Proxy> surfs;
static vector<vector<int32_t>> regions;
static double g_tolerance = 1.0e-6;
static double g_root_volume = 1.0;
static int g_max_slices = 1 << 16;

struct Stats {
  vector<double> lower, slack;
  long nodes {0}, integrated {0};
};

static BoxSense classify(int i, const OctBox& box)
{
  const Proxy& pr = surfs[i];
  double lo, hi;
  if (pr.kind == Proxy::Kind::QUADRIC) {
    form_range(pr.q, box, lo, hi);
    return from_range(lo, hi);
  }
  if (pr.kind == Proxy::Kind::TORUS && !box.rotated) {
    torus_range(pr.ax, pr.tc, pr.A, pr.B, pr.C, box, lo, hi);
    return from_range(lo, hi);
  }
  return BoxSense::BOTH;
}

static bool try_integrate(const OctBox& box,
  const std::unordered_map<int32_t, BoxSense>& memo, Stats& st)
{
  vector<int32_t> strad;
  for (auto& kv : memo)
    if (kv.second == BoxSense::BOTH)
      strad.push_back(kv.first);
  // The all-planes path costs 2^n polytope volumes, the curved paths 2^n
  // annulus evaluations; 8 is where that stops being worth it. Checked again
  // per path below, since the curved paths are the more expensive per piece.
  if (strad.empty() || strad.size() > 8)
    return false;
  std::sort(strad.begin(), strad.end());
  int n = strad.size();

  auto sense_of = [&](int32_t i) {
    auto it = memo.find(i);
    return it == memo.end() ? BoxSense::UNKNOWN : it->second;
  };

  struct Piece {
    double vol;
    vector<BoxSense> signs;
  };
  vector<Piece> pieces;
  double extra_slack = 0.0;

  vector<Position> pl_n(n);
  vector<double> pl_d(n);
  vector<char> is_plane(n, 0);

  struct CylGroup {
    double c1, c2, r_max {0};
    vector<int> members;
  };
  vector<CylGroup> groups;
  int cyl_axis = -1, sym = -1;
  AxisSymProfile prof;
  vector<int> sym_group;
  vector<double> sym_K;
  bool all_planes = true;

  for (int i = 0; i < n; ++i) {
    const Proxy& pr = surfs[strad[i]];
    if (pr.kind == Proxy::Kind::QUADRIC && as_plane(pr.q, pl_n[i], pl_d[i])) {
      is_plane[i] = 1;
      continue;
    }
    all_planes = false;
    int a;
    double c1, c2, r;
    if (pr.kind == Proxy::Kind::QUADRIC &&
        as_axis_cylinder(pr.q, a, c1, c2, r)) {
      if (cyl_axis < 0)
        cyl_axis = a;
      else if (a != cyl_axis)
        return false;
      int g = -1;
      for (size_t k = 0; k < groups.size(); ++k)
        if (groups[k].c1 == c1 && groups[k].c2 == c2)
          g = k;
      if (g < 0) {
        groups.push_back({c1, c2, 0.0, {}});
        g = groups.size() - 1;
      }
      groups[g].members.push_back(i);
      groups[g].r_max = std::max(groups[g].r_max, r);
      continue;
    }
    if (pr.kind == Proxy::Kind::TORUS) {
      if (sym >= 0)
        return false; // a torus cannot join a concentric group
      prof = torus_profile(pr.ax, pr.tc, pr.A, pr.B, pr.C);
      sym = i;
      sym_group.push_back(i);
      sym_K.push_back(0.0);
      continue;
    }
    if (pr.kind == Proxy::Kind::QUADRIC) {
      AxisSymProfile cand;
      if (!as_axis_symmetric(pr.q, cand))
        return false;
      if (sym < 0) {
        prof = cand;
        sym = i;
      } else if (!same_revolved_geometry(prof, cand))
        return false;
      sym_group.push_back(i);
      sym_K.push_back(cand.K);
      continue;
    }
    return false;
  }
  if (sym >= 0 && sym_group.size() > 1 &&
      prof.kind == AxisSymProfile::Kind::TORUS)
    return false;
  if (sym >= 0 && !groups.empty())
    return false;
  for (size_t a = 0; a < groups.size(); ++a)
    for (size_t b = a + 1; b < groups.size(); ++b) {
      double dx = groups[a].c1 - groups[b].c1, dy = groups[a].c2 - groups[b].c2;
      if (std::sqrt(dx * dx + dy * dy) < groups[a].r_max + groups[b].r_max)
        return false;
    }

  if (all_planes) {
    for (int mask = 0; mask < (1 << n); ++mask) {
      vector<Position> nn(n);
      vector<double> dd(n);
      vector<BoxSense> sg(n);
      for (int i = 0; i < n; ++i) {
        bool neg = ((mask >> i) & 1) == 0;
        nn[i] = neg ? pl_n[i] : -1.0 * pl_n[i];
        dd[i] = neg ? pl_d[i] : -pl_d[i];
        sg[i] = neg ? BoxSense::NEGATIVE : BoxSense::POSITIVE;
      }
      double v;
      if (n == 1) {
        if (!box_halfspace_volume(box, nn[0], dd[0], v))
          return false;
      } else
        v = box_planes_volume(box, nn, dd);
      if (v > 0.0)
        pieces.push_back({v, sg});
    }
  } else {
    if (n > 6)
      return false;
    int axis = (cyl_axis >= 0) ? cyl_axis : (sym >= 0 ? prof.axis : -1);
    if (axis < 0)
      return false;
    int r1 = (axis + 1) % 3, r2 = (axis + 2) % 3;

    vector<double> base_px, base_py;
    double w_lo, w_hi;
    if (!box_axis_section(box, axis, base_px, base_py, w_lo, w_hi))
      return false;

    auto comp = [](const Position& v, int k) {
      return k == 0 ? v.x : (k == 1 ? v.y : v.z);
    };

    vector<int> axial_planes, radial_planes;
    for (int i = 0; i < n; ++i) {
      if (!is_plane[i])
        continue;
      double na = comp(pl_n[i], axis);
      double n1 = comp(pl_n[i], r1), n2 = comp(pl_n[i], r2);
      if (n1 == 0.0 && n2 == 0.0)
        axial_planes.push_back(i);
      else if (na == 0.0)
        radial_planes.push_back(i);
      else
        return false;
    }
    int na_pl = axial_planes.size(), nr_pl = radial_planes.size(), n_cyl = 0;
    for (auto& g : groups)
      n_cyl += g.members.size();
    if ((1 << (na_pl + nr_pl + n_cyl)) > 512)
      return false;

    for (int am = 0; am < (1 << na_pl); ++am) {
      double lw = w_lo, hw = w_hi;
      vector<BoxSense> axial_signs(n, BoxSense::UNKNOWN);
      bool empty = false;
      for (int j = 0; j < na_pl; ++j) {
        int i = axial_planes[j];
        double na = comp(pl_n[i], axis);
        double value = pl_d[i] / na;
        bool below = na > 0.0;
        bool take_low = ((am >> j) & 1) == 0;
        if (take_low)
          hw = std::min(hw, value);
        else
          lw = std::max(lw, value);
        if (hw <= lw) {
          empty = true;
          break;
        }
        axial_signs[i] =
          (take_low == below) ? BoxSense::NEGATIVE : BoxSense::POSITIVE;
      }
      if (empty)
        continue;
      double axial = hw - lw;

      for (int rm = 0; rm < (1 << nr_pl); ++rm) {
        vector<double> px = base_px, py = base_py;
        vector<BoxSense> signs = axial_signs;
        for (int j = 0; j < nr_pl; ++j) {
          int i = radial_planes[j];
          double a = comp(pl_n[i], r1), b = comp(pl_n[i], r2), c = pl_d[i];
          bool neg = ((rm >> j) & 1) == 0;
          if (neg) {
            clip_poly(px, py, a, b, c);
            signs[i] = BoxSense::NEGATIVE;
          } else {
            clip_poly(px, py, -a, -b, -c);
            signs[i] = BoxSense::POSITIVE;
          }
        }
        if (px.size() < 3)
          continue;
        double cross_area = std::abs(poly_area(px, py));
        if (cross_area <= 0.0)
          continue;
        double sub_vol = cross_area * axial;

        if (!groups.empty()) {
          for (int cm = 0; cm < (1 << n_cyl); ++cm) {
            vector<BoxSense> sg = signs;
            int bit = 0, n_bounded = 0, i_bounded = -1;
            vector<double> gin(groups.size(), INFINITY),
              gout(groups.size(), 0.0);
            bool dead = false;
            for (size_t g = 0; g < groups.size() && !dead; ++g) {
              for (int i : groups[g].members) {
                int aa;
                double d1, d2, r;
                as_axis_cylinder(surfs[strad[i]].q, aa, d1, d2, r);
                if (((cm >> bit) & 1) == 0) {
                  gin[g] = std::min(gin[g], r);
                  sg[i] = BoxSense::NEGATIVE;
                } else {
                  gout[g] = std::max(gout[g], r);
                  sg[i] = BoxSense::POSITIVE;
                }
                ++bit;
              }
              if (gout[g] >= gin[g])
                dead = true;
              if (!std::isinf(gin[g])) {
                ++n_bounded;
                i_bounded = g;
              }
            }
            if (dead || n_bounded > 1)
              continue;
            double area;
            if (n_bounded == 1) {
              auto& g = groups[i_bounded];
              area = circle_poly_area(g.c1, g.c2, gin[i_bounded], px, py) -
                     circle_poly_area(g.c1, g.c2, gout[i_bounded], px, py);
            } else {
              area = cross_area;
              for (size_t g = 0; g < groups.size(); ++g)
                area -=
                  circle_poly_area(groups[g].c1, groups[g].c2, gout[g], px, py);
            }
            if (area <= 0.0)
              continue;
            pieces.push_back({area * axial, sg});
          }
        } else if (sym >= 0) {
          // Bound each member's disk integral ONCE, then build every sign
          // combination by differencing. A combination is the annulus between
          // the binding "inside" surface (largest K, smallest radius) and the
          // binding "outside" one (smallest K, largest radius), and since
          // integral bounds subtract exactly the way per-slice bounds do, this
          // gives the identical bracket for ns evaluations instead of 2^ns.
          int ns = (int)sym_group.size();
          bool torus = prof.kind == AxisSymProfile::Kind::TORUS;
          double target = g_tolerance * sub_vol / g_root_volume;
          vector<double> mlo(ns), mhi(ns);
          for (int j = 0; j < ns; ++j) {
            AxisSymProfile pj = prof;
            if (!torus)
              pj.K = sym_K[j];
            int N = 32;
            axis_sym_slice_bounds_poly(pj, lw, hw, px, py, N, mlo[j], mhi[j]);
            double spread = mhi[j] - mlo[j];
            if (spread > target && target > 0.0) {
              double want = 32.0 * spread / target;
              while (N < want && N < g_max_slices)
                N *= 2;
              axis_sym_slice_bounds_poly(pj, lw, hw, px, py, N, mlo[j], mhi[j]);
            }
          }

          double sum_lo = 0.0;
          for (int sm = 0; sm < (1 << ns); ++sm) {
            int ia = -1, ib = -1;
            vector<BoxSense> sg = signs;
            for (int j = 0; j < ns; ++j) {
              bool inside = ((sm >> j) & 1) == 0;
              sg[sym_group[j]] =
                inside ? BoxSense::NEGATIVE : BoxSense::POSITIVE;
              if (inside) {
                if (ia < 0 || sym_K[j] > sym_K[ia])
                  ia = j;
              } else {
                if (ib < 0 || sym_K[j] < sym_K[ib])
                  ib = j;
              }
            }
            if (ia >= 0 && ib >= 0 && sym_K[ia] >= sym_K[ib])
              continue; // inner radius would exceed the outer: empty
            double a_lo = (ia >= 0) ? mlo[ia] : sub_vol;
            double a_hi = (ia >= 0) ? mhi[ia] : sub_vol;
            double b_lo = (ib >= 0) ? mlo[ib] : 0.0;
            double b_hi = (ib >= 0) ? mhi[ib] : 0.0;
            double slo = a_lo - b_hi;
            if (slo > 0.0) {
              pieces.push_back({slo, sg});
              sum_lo += slo;
            }
          }
          // Whatever the bounds failed to account for is slack, which also
          // keeps the tiling check below exact.
          extra_slack += sub_vol - sum_lo;
        } else {
          pieces.push_back({sub_vol, signs});
        }
      }
    }
  }

  if (pieces.empty())
    return false;

  double sum = 0.0;
  for (auto& pc : pieces)
    sum += pc.vol;
  if (std::abs(sum + extra_slack - box.volume()) > 1.0e-9 * box.volume())
    return false;

  vector<int> owner(pieces.size(), -1);
  for (size_t p = 0; p < pieces.size(); ++p) {
    auto definite = [&](int32_t i) -> BoxSense {
      for (int k = 0; k < n; ++k)
        if (strad[k] == i)
          return pieces[p].signs[k];
      return sense_of(i);
    };
    int hits = 0;
    for (int c = 0; c < (int)regions.size(); ++c)
      if (evaluate_region_tri(regions[c], true, definite) == Tri::kTrue) {
        owner[p] = c;
        ++hits;
      }
    if (hits > 1)
      return false;
  }
  for (size_t p = 0; p < pieces.size(); ++p)
    if (owner[p] >= 0)
      st.lower[owner[p]] += pieces[p].vol;
  for (size_t c = 0; c < regions.size(); ++c)
    st.slack[c] += extra_slack + INTEGRATOR_GUARD * box.volume();
  ++st.integrated;
  return true;
}

static void descend(const OctBox& box, int depth, int max_depth, Stats& st)
{
  ++st.nodes;
  std::unordered_map<int32_t, BoxSense> memo;
  auto sense = [&](int32_t i) {
    auto it = memo.find(i);
    if (it != memo.end())
      return it->second;
    BoxSense s = classify(i, box);
    memo.emplace(i, s);
    return s;
  };
  int n_true = 0, i_true = -1;
  bool maybe = false;
  for (int c = 0; c < (int)regions.size(); ++c) {
    Tri t = evaluate_region_tri(regions[c], true, sense);
    if (t == Tri::kTrue) {
      ++n_true;
      i_true = c;
    } else if (t == Tri::kMaybe)
      maybe = true;
  }
  if (n_true == 0 && !maybe)
    return;
  if (n_true == 1 && !maybe) {
    st.lower[i_true] += box.volume();
    return;
  }
  if (try_integrate(box, memo, st))
    return;
  if (depth >= max_depth) {
    for (int c = 0; c < (int)regions.size(); ++c)
      if (evaluate_region_tri(regions[c], true, sense) != Tri::kFalse)
        st.slack[c] += box.volume();
    return;
  }
  for (int i = 0; i < 8; ++i)
    descend(box.child(i), depth + 1, max_depth, st);
}

static Stats run(const OctBox& root, int max_depth)
{
  Stats st;
  st.lower.assign(regions.size(), 0.0);
  st.slack.assign(regions.size(), 0.0);
  g_root_volume = root.volume();
  descend(root, 0, max_depth, st);
  return st;
}

static void report(const char* name, const Stats& st, int cell, double exact)
{
  double lo = st.lower[cell], hi = lo + st.slack[cell];
  bool ok = lo <= exact + 1e-12 && exact <= hi + 1e-12;
  std::printf("  %-34s [%.9f, %.9f]  half-width %.2e  %6ld nodes  %s\n", name,
    lo, hi, 0.5 * st.slack[cell], st.nodes,
    ok ? "brackets" : "*** VIOLATION ***");
  OCT_CHECK(ok, "%s: %.10f not in [%.10f, %.10f]", name, exact, lo, hi);
}

//==============================================================================

static QuadricForm plane_q(int ax, double v)
{
  QuadricForm q;
  (ax == 0 ? q.b.x : (ax == 1 ? q.b.y : q.b.z)) = 1.0;
  q.c = -v;
  return q;
}
static QuadricForm oblique_q(Position n, double d)
{
  QuadricForm q;
  q.b = n;
  q.c = -d;
  return q;
}
static QuadricForm zcyl_q(double x0, double y0, double r)
{
  QuadricForm q;
  q.A[0] = q.A[1] = 1.0;
  q.b = {-2 * x0, -2 * y0, 0.0};
  q.c = x0 * x0 + y0 * y0 - r * r;
  return q;
}
static void push_q(const QuadricForm& q)
{
  Proxy p;
  p.kind = Proxy::Kind::QUADRIC;
  p.q = q;
  surfs.push_back(p);
}
static vector<int32_t> with(vector<int32_t> a, const vector<int32_t>& b)
{
  for (int32_t t : b)
    a.push_back(t);
  return a;
}

//------------------------------------------------------------------------------
// 1. Torus. Previously had no integrator at all.
//------------------------------------------------------------------------------
static void test_torus()
{
  std::printf("torus (was: no integrator, refinement only)\n");
  surfs.clear();
  regions.clear();
  double A = 5.0, B = 1.0, C = 1.5;
  Proxy t;
  t.kind = Proxy::Kind::TORUS;
  t.ax = 2;
  t.tc = {0, 0, 0};
  t.A = A;
  t.B = B;
  t.C = C;
  surfs.push_back(t);              // 1
  for (int ax = 0; ax < 3; ++ax) { // 2..7
    push_q(plane_q(ax, ax == 2 ? -2.0 : -8.0));
    push_q(plane_q(ax, ax == 2 ? 2.0 : 8.0));
  }
  vector<int32_t> box {2, -3, 4, -5, 6, -7};
  regions = {with({-1}, box), with({1}, box)};
  OctBox root = OctBox::from_aabb({-8, -8, -2}, {8, 8, 2});
  report("torus 2 pi^2 A B C", run(root, 6), 0, 2 * M_PI * M_PI * A * B * C);
}

//------------------------------------------------------------------------------
// 2. Bundle of disjoint parallel cylinders -- the paper's own integrator.
//------------------------------------------------------------------------------
static void test_cylinder_bundle()
{
  std::printf("disjoint cylinder bundle (was: non-coaxial -> refuse)\n");
  surfs.clear();
  regions.clear();
  double r1 = 0.5, r2 = 0.4, r3 = 0.3, H = 4.0;
  push_q(zcyl_q(-1.5, 0.0, r1));   // 1
  push_q(zcyl_q(0.0, 1.2, r2));    // 2
  push_q(zcyl_q(1.6, -0.9, r3));   // 3
  for (int ax = 0; ax < 3; ++ax) { // 4..9
    push_q(plane_q(ax, ax == 2 ? -H / 2 : -3.0));
    push_q(plane_q(ax, ax == 2 ? H / 2 : 3.0));
  }
  vector<int32_t> box {4, -5, 6, -7, 8, -9};
  regions = {
    with({-1}, box), with({-2}, box), with({-3}, box), with({1, 2, 3}, box)};
  OctBox root = OctBox::from_aabb({-3, -3, -H / 2}, {3, 3, H / 2});
  Stats st = run(root, 6);
  report("pin 1", st, 0, M_PI * r1 * r1 * H);
  report("pin 2", st, 1, M_PI * r2 * r2 * H);
  report("pin 3", st, 2, M_PI * r3 * r3 * H);
  report(
    "moderator", st, 3, 36.0 * H - M_PI * H * (r1 * r1 + r2 * r2 + r3 * r3));
}

//------------------------------------------------------------------------------
// 3. Several oblique planes -- a hexagonal prism cell.
//------------------------------------------------------------------------------
static void test_hex_prism(bool rotated)
{
  std::printf("hex prism, %s box (was: >1 plane on a rotated box -> refuse)\n",
    rotated ? "rotated" : "axis-aligned");
  surfs.clear();
  regions.clear();
  double apothem = 1.0, H = 3.0;
  for (int k = 0; k < 6; ++k) { // 1..6
    double th = k * M_PI / 3.0;
    push_q(oblique_q({std::cos(th), std::sin(th), 0.0}, apothem));
  }
  push_q(plane_q(2, -H / 2)); // 7
  push_q(plane_q(2, H / 2));  // 8
  vector<int32_t> hex {-1, -2, -3, -4, -5, -6, 7, -8};
  regions = {hex};
  OctBox root = OctBox::from_aabb({-2, -2, -H / 2}, {2, 2, H / 2});
  if (rotated) {
    // Rotate the box 20 degrees about z. The prism is unchanged, so the answer
    // must be too -- and every straddling surface is now oblique in box coords.
    double a = 20.0 * M_PI / 180.0;
    root.axis[0] = {std::cos(a), std::sin(a), 0};
    root.axis[1] = {-std::sin(a), std::cos(a), 0};
    root.axis[2] = {0, 0, 1};
    root.rotated = true;
    for (int k = 0; k < 3; ++k)
      root.half[k] = (k == 2) ? H / 2 : 2.6;
  }
  double exact = 2.0 * std::sqrt(3.0) * apothem * apothem * H;
  report("hex prism 2 sqrt(3) a^2 H", run(root, 7), 0, exact);
}

//------------------------------------------------------------------------------
// 3b. Hex cell containing a fuel pin: oblique walls parallel to the pin axis.
//------------------------------------------------------------------------------
static void test_hex_pin()
{
  std::printf(
    "hex cell + pin (was: curved surface + oblique plane -> refuse)\n");
  surfs.clear();
  regions.clear();
  double apothem = 0.7, rf = 0.45, rc = 0.55, H = 4.0;
  push_q(zcyl_q(0, 0, rf));     // 1 fuel
  push_q(zcyl_q(0, 0, rc));     // 2 clad
  for (int k = 0; k < 6; ++k) { // 3..8 hex walls
    double th = k * M_PI / 3.0;
    push_q(oblique_q({std::cos(th), std::sin(th), 0.0}, apothem));
  }
  push_q(plane_q(2, -H / 2)); // 9
  push_q(plane_q(2, H / 2));  // 10
  vector<int32_t> cell {-3, -4, -5, -6, -7, -8, 9, -10};
  regions = {with({-1}, cell), with({1, -2}, cell), with({2}, cell)};
  OctBox root = OctBox::from_aabb({-1, -1, -H / 2}, {1, 1, H / 2});
  Stats st = run(root, 7);
  double hex_area = 2.0 * std::sqrt(3.0) * apothem * apothem;
  report("fuel", st, 0, M_PI * rf * rf * H);
  report("clad", st, 1, M_PI * (rc * rc - rf * rf) * H);
  report("coolant", st, 2, (hex_area - M_PI * rc * rc) * H);
}

//------------------------------------------------------------------------------
// 3c. Pin tangent to the hex wall: the degenerate case for circle_poly_area.
//------------------------------------------------------------------------------
static void test_hex_pin_tangent()
{
  std::printf("hex cell + pin tangent to every wall (degenerate)\n");
  surfs.clear();
  regions.clear();
  double apothem = 0.7, H = 2.0;
  push_q(zcyl_q(0, 0, apothem)); // 1, exactly inscribed
  for (int k = 0; k < 6; ++k) {
    double th = k * M_PI / 3.0;
    push_q(oblique_q({std::cos(th), std::sin(th), 0.0}, apothem));
  }
  push_q(plane_q(2, -H / 2));
  push_q(plane_q(2, H / 2));
  vector<int32_t> cell {-2, -3, -4, -5, -6, -7, 8, -9};
  regions = {with({-1}, cell), with({1}, cell)};
  Stats st = run(OctBox::from_aabb({-1, -1, -H / 2}, {1, 1, H / 2}), 7);
  report("inscribed pin", st, 0, M_PI * apothem * apothem * H);
  report("hex corners", st, 1,
    (2.0 * std::sqrt(3.0) * apothem * apothem - M_PI * apothem * apothem) * H);
}

//------------------------------------------------------------------------------
// 3d. Curved surface in a box rotated ABOUT that surface's axis.
//------------------------------------------------------------------------------
static void test_rotated_box_cylinder()
{
  std::printf(
    "cylinder in a box rotated about z (was: rotated box -> refuse)\n");
  double r = 0.5, H = 3.0;
  for (double deg : {0.0, 17.0, 45.0}) {
    surfs.clear();
    regions.clear();
    push_q(zcyl_q(0, 0, r));    // 1
    push_q(plane_q(2, -H / 2)); // 2
    push_q(plane_q(2, H / 2));  // 3
    regions = {{-1, 2, -3}, {1, 2, -3}};

    OctBox root = OctBox::from_aabb({-2, -2, -H / 2}, {2, 2, H / 2});
    double a = deg * M_PI / 180.0;
    if (deg != 0.0) {
      root.axis[0] = {std::cos(a), std::sin(a), 0};
      root.axis[1] = {-std::sin(a), std::cos(a), 0};
      root.axis[2] = {0, 0, 1};
      root.rotated = true;
    }
    Stats st = run(root, 6);
    char name[64];
    std::snprintf(name, sizeof name, "cylinder, box rotated %.0f deg", deg);
    report(name, st, 0, M_PI * r * r * H);
    // The complement must make up the rest of the box exactly.
    char name2[64];
    std::snprintf(name2, sizeof name2, "  complement fills the box");
    report(name2, st, 1, root.volume() - M_PI * r * r * H);
  }
}

//------------------------------------------------------------------------------
// 3e. Quadrics with a cross term coupling the two radial axes: a tilted
//     ellipse, removable by a rotation about the symmetry axis.
//------------------------------------------------------------------------------
static void test_tilted_quadrics()
{
  std::printf("tilted quadrics (was: cross terms -> not recognised)\n");
  double H = 3.0;

  // 2x^2 + 2xy + 2y^2 - 1 <= 0 : eigenvalues 3 and 1, semi-axes 1/sqrt(3), 1.
  {
    surfs.clear();
    regions.clear();
    QuadricForm q;
    q.A[0] = 2.0;
    q.A[1] = 2.0;
    q.A[3] = 1.0;
    q.c = -1.0;
    push_q(q);
    push_q(plane_q(2, -H / 2));
    push_q(plane_q(2, H / 2));
    regions = {{-1, 2, -3}, {1, 2, -3}};
    Stats st = run(OctBox::from_aabb({-2, -2, -H / 2}, {2, 2, H / 2}), 6);
    report("tilted elliptic cylinder", st, 0, M_PI / std::sqrt(3.0) * H);
  }

  // Same block plus z^2: a tilted ellipsoid, semi-axes 1/sqrt(3), 1, 1.
  {
    surfs.clear();
    regions.clear();
    QuadricForm q;
    q.A[0] = 2.0;
    q.A[1] = 2.0;
    q.A[2] = 1.0;
    q.A[3] = 1.0;
    q.c = -1.0;
    push_q(q);
    for (int ax = 0; ax < 3; ++ax) {
      push_q(plane_q(ax, -2.0));
      push_q(plane_q(ax, 2.0));
    }
    regions = {{-1, 2, -3, 4, -5, 6, -7}, {1, 2, -3, 4, -5, 6, -7}};
    Stats st = run(OctBox::from_aabb({-2, -2, -2}, {2, 2, 2}), 6);
    report("tilted ellipsoid", st, 0,
      4.0 / 3.0 * M_PI * (1.0 / std::sqrt(3.0)) * 1.0 * 1.0);
  }

  // Tilted ellipse inside a hex cell, in a box rotated about z: everything at
  // once.
  {
    surfs.clear();
    regions.clear();
    double apothem = 1.4;
    QuadricForm q;
    q.A[0] = 2.0;
    q.A[1] = 2.0;
    q.A[3] = 1.0;
    q.c = -1.0;
    push_q(q);                    // 1
    for (int k = 0; k < 6; ++k) { // 2..7
      double th = k * M_PI / 3.0;
      push_q(oblique_q({std::cos(th), std::sin(th), 0.0}, apothem));
    }
    push_q(plane_q(2, -H / 2)); // 8
    push_q(plane_q(2, H / 2));  // 9
    vector<int32_t> cell {-2, -3, -4, -5, -6, -7, 8, -9};
    regions = {with({-1}, cell), with({1}, cell)};
    OctBox root = OctBox::from_aabb({-2, -2, -H / 2}, {2, 2, H / 2});
    double a = 31.0 * M_PI / 180.0;
    root.axis[0] = {std::cos(a), std::sin(a), 0};
    root.axis[1] = {-std::sin(a), std::cos(a), 0};
    root.axis[2] = {0, 0, 1};
    root.rotated = true;
    Stats st = run(root, 7);
    report(
      "tilted ellipse in hex, rotated box", st, 0, M_PI / std::sqrt(3.0) * H);
    report("  hex remainder", st, 1,
      (2.0 * std::sqrt(3.0) * apothem * apothem - M_PI / std::sqrt(3.0)) * H);
  }
}

//------------------------------------------------------------------------------
// 3f. Concentric spheres: a TRISO particle. This is the case refinement cannot
//     rescue, since the shells stay mutually ambiguous over a 2-D region.
//------------------------------------------------------------------------------
static void test_concentric_shells()
{
  std::printf("TRISO concentric shells (was: >1 curved surface -> refuse)\n");
  surfs.clear();
  regions.clear();
  const double R[5] {0.02125, 0.03125, 0.03525, 0.03875, 0.04275};
  for (int i = 0; i < 5; ++i) { // 1..5
    QuadricForm q;
    q.A[0] = q.A[1] = q.A[2] = 1.0;
    q.c = -R[i] * R[i];
    push_q(q);
  }
  double b = 0.05;
  for (int ax = 0; ax < 3; ++ax) { // 6..11
    push_q(plane_q(ax, -b));
    push_q(plane_q(ax, b));
  }
  vector<int32_t> box {6, -7, 8, -9, 10, -11};
  regions = {with({-1}, box), with({1, -2}, box), with({2, -3}, box),
    with({3, -4}, box), with({4, -5}, box), with({5}, box)};

  Stats st = run(OctBox::from_aabb({-b, -b, -b}, {b, b, b}), 6);
  const char* names[5] {"kernel", "buffer", "IPyC", "SiC", "OPyC"};
  for (int i = 0; i < 5; ++i) {
    double lo = i ? R[i - 1] : 0.0;
    report(
      names[i], st, i, 4.0 / 3.0 * M_PI * (R[i] * R[i] * R[i] - lo * lo * lo));
  }
  report(
    "matrix", st, 5, 8 * b * b * b - 4.0 / 3.0 * M_PI * R[4] * R[4] * R[4]);
  std::printf("  (%ld nodes, %ld integrated -- refinement alone needs the box\n"
              "   edge below the thinnest shell, %.4f cm)\n",
    st.nodes, st.integrated, R[2] - R[1]);
}

//------------------------------------------------------------------------------
// 4. Regression: pincell and sphere must still work.
//------------------------------------------------------------------------------
static void test_regression()
{
  std::printf("regression\n");
  surfs.clear();
  regions.clear();
  double rf = 0.39218, rg = 0.40005, rc = 0.45720, p = 1.26, hz = 5.0;
  push_q(zcyl_q(0, 0, rf));
  push_q(zcyl_q(0, 0, rg));
  push_q(zcyl_q(0, 0, rc));
  for (int ax = 0; ax < 3; ++ax) {
    push_q(plane_q(ax, ax == 2 ? -hz : -p / 2));
    push_q(plane_q(ax, ax == 2 ? hz : p / 2));
  }
  vector<int32_t> box {4, -5, 6, -7, 8, -9};
  regions = {
    with({-1}, box), with({1, -2}, box), with({2, -3}, box), with({3}, box)};
  Stats st =
    run(OctBox::from_aabb({-p / 2, -p / 2, -hz}, {p / 2, p / 2, hz}), 6);
  report("pincell fuel", st, 0, M_PI * rf * rf * 2 * hz);
  report("pincell clad", st, 2, M_PI * (rc * rc - rg * rg) * 2 * hz);

  surfs.clear();
  regions.clear();
  QuadricForm sph;
  sph.A[0] = sph.A[1] = sph.A[2] = 1.0;
  sph.c = -9.0;
  push_q(sph);
  for (int ax = 0; ax < 3; ++ax) {
    push_q(plane_q(ax, -4.0));
    push_q(plane_q(ax, 4.0));
  }
  vector<int32_t> b2 {2, -3, 4, -5, 6, -7};
  regions = {with({-1}, b2), with({1}, b2)};
  g_max_slices = 1 << 16;
  report("sphere r=3", run(OctBox::from_aabb({-4, -4, -4}, {4, 4, 4}), 6), 0,
    4.0 / 3.0 * M_PI * 27.0);
  g_max_slices = 1 << 16;
}

//------------------------------------------------------------------------------
// 5. How badly does refinement handle a box the integrators refuse?
//------------------------------------------------------------------------------
//
// Two intersecting spheres. Neither is concentric with the other, so no
// integrator fires where both straddle, and those boxes fall back to the Box
// integrator. The question is what that costs.
static void test_two_curved_convergence()
{
  std::printf("two intersecting spheres: refinement fallback convergence\n");
  double R = 1.0, d = 0.8; // centres at +-d/2 on x, so they overlap
  double prev_slack = 0.0;
  long prev_nodes = 0;

  // Analytic union volume: two balls minus the lens of intersection.
  double cap_h = R - d / 2.0;
  double lens = 2.0 * (M_PI * cap_h * cap_h * (3.0 * R - cap_h) / 3.0);
  double exact = 2.0 * (4.0 / 3.0 * M_PI * R * R * R) - lens;

  std::printf("%6s %10s %14s %14s   %s\n", "depth", "nodes", "slack",
    "half-width", "per level");
  for (int depth = 2; depth <= 8; ++depth) {
    surfs.clear();
    regions.clear();
    QuadricForm s1;
    s1.A[0] = s1.A[1] = s1.A[2] = 1.0;
    s1.b = {-2 * (-d / 2), 0, 0};
    s1.c = (d / 2) * (d / 2) - R * R;
    QuadricForm s2;
    s2.A[0] = s2.A[1] = s2.A[2] = 1.0;
    s2.b = {-2 * (d / 2), 0, 0};
    s2.c = (d / 2) * (d / 2) - R * R;
    push_q(s1);
    push_q(s2);
    for (int ax = 0; ax < 3; ++ax) {
      push_q(plane_q(ax, -2.0));
      push_q(plane_q(ax, 2.0));
    }
    vector<int32_t> box {3, -4, 5, -6, 7, -8};
    // union of the two balls, as a complement of the intersection of exteriors
    regions = {with({-1}, box), with({1, -2}, box), with({1, 2}, box)};

    Stats st = run(OctBox::from_aabb({-2, -2, -2}, {2, 2, 2}), depth);
    // union volume = ball1 + (ball2 minus ball1)
    double lo = st.lower[0] + st.lower[1];
    double hi = lo + st.slack[0] + st.slack[1];
    bool ok = lo <= exact + 1e-12 && exact <= hi + 1e-12;
    double slack = st.slack[0] + st.slack[1];
    std::printf("%6d %10ld %14.6e %14.6e", depth, st.nodes, slack, 0.5 * slack);
    if (prev_slack > 0)
      std::printf("   slack x%.2f  nodes x%.2f", slack / prev_slack,
        double(st.nodes) / prev_nodes);
    std::printf("  %s\n", ok ? "" : "*** VIOLATION ***");
    OCT_CHECK(ok, "two spheres depth %d: %.9f not in [%.9f, %.9f]", depth,
      exact, lo, hi);
    prev_slack = slack;
    prev_nodes = st.nodes;
  }
  std::printf("  (slack x0.25 and nodes x2 per level would mean the ambiguity\n"
              "   has retreated to the 1-D intersection curve)\n");
}

static int run_all()
{
  test_torus();
  test_cylinder_bundle();
  test_hex_prism(false);
  test_hex_prism(true);
  test_hex_pin();
  test_hex_pin_tangent();
  test_rotated_box_cylinder();
  test_tilted_quadrics();
  test_concentric_shells();
  test_regression();
  test_two_curved_convergence();
  if (failures == 0) {
    std::printf("\nall integrator checks passed\n");
    return 0;
  }
  std::printf("\n%d FAILURES\n", failures);
  return 1;
}

TEST_CASE("volume_octree: base-case integrators, end to end", "[volume_octree]")
{
  // The body reports its own diagnostics on stdout and returns a failure count;
  // run with -s to see the tables even when everything passes.
  REQUIRE(run_all() == 0);
}
