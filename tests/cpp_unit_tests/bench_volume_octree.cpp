//! Measures how the bracket tightens with octree depth, so the cost of a given
//! tolerance can be extrapolated rather than guessed.
//!
//! Standalone: a hand-coded pincell using volume_octree_math.h directly, no
//! OpenMC build needed.
//!
//!   g++ -std=c++17 -O2 -I include -I <fmt> \
//!       tests/cpp_unit_tests/bench_volume_octree.cpp src/position.cpp -o bench

#include "openmc/volume_octree_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>

using namespace openmc;

// Pincell: three concentric ZCylinders in a 1.26 x 1.26 x 10 box.
static const double R_F = 0.39218, R_G = 0.40005, R_C = 0.45720;
static const double PITCH = 1.26, HALF_Z = 5.0;

static vector<QuadricForm> surfs;
static vector<vector<int32_t>> regions;

static void build()
{
  auto zcyl = [](double r) {
    QuadricForm q;
    q.A[0] = q.A[1] = 1.0;
    q.c = -r * r;
    return q;
  };
  auto plane = [](int ax, double v) {
    QuadricForm q;
    (ax == 0 ? q.b.x : (ax == 1 ? q.b.y : q.b.z)) = 1.0;
    q.c = -v;
    return q;
  };
  surfs = {zcyl(R_F), zcyl(R_G), zcyl(R_C), plane(0, -PITCH / 2),
    plane(0, PITCH / 2), plane(1, -PITCH / 2), plane(1, PITCH / 2),
    plane(2, -HALF_Z), plane(2, HALF_Z)};

  vector<int32_t> box {4, -5, 6, -7, 8, -9};
  auto with_box = [&](vector<int32_t> v) {
    for (int32_t t : box)
      v.push_back(t);
    return v;
  };
  regions = {
    with_box({-1}), with_box({1, -2}), with_box({2, -3}), with_box({3})};
}

static bool g_integrators = false;
static double g_tolerance = 1.0e-6;
static double g_root_volume = 1.0;

struct Stats {
  double lower[4] {};
  double slack[4] {};
  long nodes {0};
  long leaves {0};
};

//! Same logic as OctreeVolumeCalculation::try_integrate, reduced to the
//! standalone setting: axis-aligned planes cut the box into sub-boxes, coaxial
//! axis-aligned cylinders reduce to annuli, and a lone plane of any
//! orientation is handled by the halfspace formula.
static bool try_integrate(const OctBox& box,
  const std::unordered_map<int32_t, BoxSense>& memo, Stats& st)
{
  vector<int32_t> strad;
  for (auto& kv : memo)
    if (kv.second == BoxSense::BOTH)
      strad.push_back(kv.first);
  if (strad.empty() || strad.size() > 8)
    return false;
  std::sort(strad.begin(), strad.end());
  int n = strad.size();

  struct Piece {
    double vol;
    vector<BoxSense> signs;
  };
  vector<Piece> pieces;
  double extra_slack = 0.0;
  Position nn_tmp;
  double dd_tmp;
  AxisSymProfile prof;
  int sym = -1;

  vector<int> pl, cy;
  vector<int> pax(n, -1), cax(n, -1);
  vector<double> pval(n), crad(n);
  vector<char> pbel(n, 0);
  int caxis = -1;
  double cc1 = 0, cc2 = 0;

  for (int i = 0; i < n; ++i) {
    const QuadricForm& q = surfs[strad[i]];
    int a;
    double v, c1, c2, r;
    bool below;
    if (!box.rotated && as_axis_plane(q, a, v, below)) {
      pax[i] = a;
      pval[i] = v;
      pbel[i] = below;
      pl.push_back(i);
    } else if (!box.rotated && as_axis_cylinder(q, a, c1, c2, r)) {
      if (caxis < 0) {
        caxis = a;
        cc1 = c1;
        cc2 = c2;
      } else if (a != caxis || c1 != cc1 || c2 != cc2)
        return false;
      cax[i] = a;
      crad[i] = r;
      cy.push_back(i);
    } else if (!box.rotated && sym < 0 && as_axis_symmetric(q, prof)) {
      sym = i;
    } else if (n == 1 && as_plane(q, nn_tmp, dd_tmp)) {
      double vneg;
      if (!box_halfspace_volume(box, nn_tmp, dd_tmp, vneg))
        return false;
      pieces.push_back({vneg, {BoxSense::NEGATIVE}});
      pieces.push_back({box.volume() - vneg, {BoxSense::POSITIVE}});
    } else {
      return false;
    }
  }

  if (pieces.empty()) {
    if (box.rotated)
      return false;
    int np = pl.size(), nc = cy.size();
    if (sym >= 0 && nc > 0)
      return false;
    if ((1 << (np + nc)) > 256)
      return false;
    double lo0[3] {box.center.x - box.half[0], box.center.y - box.half[1],
      box.center.z - box.half[2]};
    double hi0[3] {box.center.x + box.half[0], box.center.y + box.half[1],
      box.center.z + box.half[2]};
    for (int pm = 0; pm < (1 << np); ++pm) {
      double lo[3] {lo0[0], lo0[1], lo0[2]}, hi[3] {hi0[0], hi0[1], hi0[2]};
      vector<BoxSense> signs(n, BoxSense::UNKNOWN);
      bool empty = false;
      for (int j = 0; j < np; ++j) {
        int i = pl[j], a = pax[i];
        bool low = ((pm >> j) & 1) == 0;
        if (low)
          hi[a] = std::min(hi[a], pval[i]);
        else
          lo[a] = std::max(lo[a], pval[i]);
        if (hi[a] <= lo[a]) {
          empty = true;
          break;
        }
        signs[i] =
          (low == (pbel[i] != 0)) ? BoxSense::NEGATIVE : BoxSense::POSITIVE;
      }
      if (empty)
        continue;
      double sub_vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
      if (nc == 0) {
        if (sym >= 0) {
          OctBox sb =
            OctBox::from_aabb({lo[0], lo[1], lo[2]}, {hi[0], hi[1], hi[2]});
          double target = g_tolerance * sub_vol / g_root_volume;
          double lo32, hi32;
          axis_sym_slice_bounds(sb, prof, 32, lo32, hi32);
          double spread = hi32 - lo32;
          int N = 32;
          if (spread > target && target > 0.0) {
            double want = 32.0 * spread / target;
            while (N < want && N < (1 << 26))
              N *= 2;
          }
          double slo = lo32, shi = hi32;
          if (N > 32)
            axis_sym_slice_bounds(sb, prof, N, slo, shi);
          vector<BoxSense> sg = signs;
          sg[sym] = BoxSense::NEGATIVE;
          pieces.push_back({slo, sg});
          sg[sym] = BoxSense::POSITIVE;
          pieces.push_back({sub_vol - shi, sg});
          extra_slack += shi - slo;
        } else {
          pieces.push_back({sub_vol, signs});
        }
        continue;
      }
      int a = caxis, r1 = (a + 1) % 3, r2 = (a + 2) % 3;
      double axial = hi[a] - lo[a];
      for (int cm = 0; cm < (1 << nc); ++cm) {
        double rin = INFINITY, rout = 0.0;
        vector<BoxSense> sg = signs;
        for (int j = 0; j < nc; ++j) {
          int i = cy[j];
          if (((cm >> j) & 1) == 0) {
            rin = std::min(rin, crad[i]);
            sg[i] = BoxSense::NEGATIVE;
          } else {
            rout = std::max(rout, crad[i]);
            sg[i] = BoxSense::POSITIVE;
          }
        }
        if (rout >= rin)
          continue;
        double outer = std::isinf(rin) ? (hi[r1] - lo[r1]) * (hi[r2] - lo[r2])
                                       : circle_rect_area(cc1, cc2, rin, lo[r1],
                                           hi[r1], lo[r2], hi[r2]);
        double inner = rout == 0.0 ? 0.0
                                   : circle_rect_area(cc1, cc2, rout, lo[r1],
                                       hi[r1], lo[r2], hi[r2]);
        if (outer - inner <= 0.0)
          continue;
        pieces.push_back({(outer - inner) * axial, sg});
      }
    }
  }
  if (pieces.empty())
    return false;

  for (auto& pc : pieces) {
    auto definite = [&](int32_t i) -> BoxSense {
      for (int k = 0; k < n; ++k)
        if (strad[k] == i)
          return pc.signs[k];
      auto it = memo.find(i);
      return it == memo.end() ? BoxSense::UNKNOWN : it->second;
    };
    int owner = -1, hits = 0;
    for (int c = 0; c < (int)regions.size(); ++c)
      if (evaluate_region_tri(regions[c], true, definite) == Tri::kTrue) {
        owner = c;
        ++hits;
      }
    if (hits > 1)
      return false;
    if (owner >= 0)
      st.lower[owner] += pc.vol;
  }
  for (int c = 0; c < (int)regions.size(); ++c)
    st.slack[c] += extra_slack + INTEGRATOR_GUARD * box.volume();
  ++st.leaves;
  return true;
}

static void descend(
  const OctBox& box, int depth, int max_depth, bool anisotropic, Stats& st)
{
  ++st.nodes;

  std::unordered_map<int32_t, BoxSense> memo;
  auto sense = [&](int32_t i) {
    auto it = memo.find(i);
    if (it != memo.end())
      return it->second;
    double lo, hi;
    form_range(surfs[i], box, lo, hi);
    BoxSense s = from_range(lo, hi);
    memo.emplace(i, s);
    return s;
  };

  int n_true = 0, i_true = -1;
  bool any_maybe = false;
  for (int c = 0; c < (int)regions.size(); ++c) {
    Tri t = evaluate_region_tri(regions[c], true, sense);
    if (t == Tri::kTrue) {
      ++n_true;
      i_true = c;
    } else if (t == Tri::kMaybe) {
      any_maybe = true;
    }
  }

  if (n_true == 0 && !any_maybe)
    return;

  if (n_true == 1 && !any_maybe) {
    st.lower[i_true] += box.volume();
    ++st.leaves;
    return;
  }

  if (g_integrators && try_integrate(box, memo, st))
    return;

  if (depth >= max_depth) {
    for (int c = 0; c < (int)regions.size(); ++c) {
      Tri t = evaluate_region_tri(regions[c], true, sense);
      if (t != Tri::kFalse)
        st.slack[c] += box.volume();
    }
    ++st.leaves;
    return;
  }

  // Which axes are worth splitting? An axis no straddling surface depends on
  // produces two children with identical classifications: pure waste.
  bool split[3] {true, true, true};
  if (anisotropic) {
    for (int k = 0; k < 3; ++k) {
      bool needed = false;
      for (auto& kv : memo) {
        const QuadricForm& q = surfs[kv.first];
        if (kv.second == BoxSense::BOTH &&
            !axis_irrelevant(q, box, k, form_gradient(q, box))) {
          needed = true;
          break;
        }
      }
      split[k] = needed;
    }
    if (!split[0] && !split[1] && !split[2])
      split[0] = split[1] = split[2] = true; // shouldn't happen; stay safe
  }

  int mask = 0;
  for (int k = 0; k < 3; ++k)
    if (split[k])
      mask |= (1 << k);

  int n_child = OctBox::n_children(mask);
  for (int i = 0; i < n_child; ++i)
    descend(box.child(i, mask), depth + 1, max_depth, anisotropic, st);
}

static void run(bool anisotropic)
{
  std::printf(
    "\n%s subdivision\n", anisotropic ? "Anisotropic" : "Full octree");
  std::printf("%6s %14s %14s %14s %10s\n", "depth", "nodes", "slack(fuel)",
    "half-width", "time(s)");

  double exact = M_PI * R_F * R_F * 2 * HALF_Z;
  double prev_slack = 0.0;
  long prev_nodes = 0;

  for (int d = 1; d <= 16; ++d) {
    Stats st;
    OctBox root = OctBox::from_aabb(
      {-PITCH / 2, -PITCH / 2, -HALF_Z}, {PITCH / 2, PITCH / 2, HALF_Z});
    g_root_volume = root.volume();

    auto t0 = std::chrono::steady_clock::now();
    descend(root, 0, d, anisotropic, st);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    // Sanity: the bracket must contain the analytic answer.
    if (!(st.lower[0] <= exact + 1e-12 &&
          exact <= st.lower[0] + st.slack[0] + 1e-12)) {
      std::printf("  BRACKET VIOLATION at depth %d\n", d);
    }

    std::printf("%6d %14ld %14.6e %14.6e %10.3f  err %+.3e", d, st.nodes,
      st.slack[0], 0.5 * st.slack[0], secs,
      st.lower[0] + 0.5 * st.slack[0] - exact);
    if (prev_slack > 0)
      std::printf("   slack x%.2f  nodes x%.2f", st.slack[0] / prev_slack,
        double(st.nodes) / prev_nodes);
    std::printf("\n");
    prev_slack = st.slack[0];
    prev_nodes = st.nodes;

    if (secs > 20.0) {
      std::printf("  (stopping: getting slow)\n");
      break;
    }
  }
}

//! A sphere has no irrelevant axis anywhere, so it is the honest general case.
static void run_sphere()
{
  surfs.clear();
  regions.clear();
  QuadricForm sph;
  sph.A[0] = sph.A[1] = sph.A[2] = 1.0;
  double R = 3.0;
  sph.c = -R * R;
  surfs.push_back(sph);
  auto plane = [](int ax, double v) {
    QuadricForm q;
    (ax == 0 ? q.b.x : (ax == 1 ? q.b.y : q.b.z)) = 1.0;
    q.c = -v;
    return q;
  };
  for (int ax = 0; ax < 3; ++ax) {
    surfs.push_back(plane(ax, -4.0));
    surfs.push_back(plane(ax, 4.0));
  }
  vector<int32_t> box {2, -3, 4, -5, 6, -7};
  vector<int32_t> in {-1}, out {1};
  for (int32_t t : box) {
    in.push_back(t);
    out.push_back(t);
  }
  regions = {in, out};

  std::printf("\nSphere r=3 in an 8 cm cube (no extrusion to exploit)\n");
  std::printf("%6s %14s %14s %14s %10s\n", "depth", "nodes", "slack",
    "half-width", "time(s)");
  double prev_slack = 0.0;
  long prev_nodes = 0;
  for (int d = 1; d <= 4; ++d) {
    Stats st;
    OctBox root = OctBox::from_aabb({-4., -4., -4.}, {4., 4., 4.});
    g_root_volume = root.volume();
    auto t0 = std::chrono::steady_clock::now();
    descend(root, 0, d, true, st);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double ex = 4.0 / 3.0 * M_PI * 27.0;
    bool ok =
      st.lower[0] <= ex + 1e-12 && ex <= st.lower[0] + st.slack[0] + 1e-12;
    std::printf("%6d %14ld %14.6e %14.6e %10.3f  err %+.3e %s", d, st.nodes,
      st.slack[0], 0.5 * st.slack[0], secs,
      st.lower[0] + 0.5 * st.slack[0] - ex,
      ok ? "" : "*** BRACKET VIOLATION ***");
    if (prev_slack > 0)
      std::printf("   slack x%.2f  nodes x%.2f", st.slack[0] / prev_slack,
        double(st.nodes) / prev_nodes);
    std::printf("\n");
    prev_slack = st.slack[0];
    prev_nodes = st.nodes;
    if (secs > 20.0)
      break;
  }
}

//! Mirror OctreeVolumeCalculation::run(): iterative deepening from
//! start_depth_, stopping on the measured half-width, and report the CUMULATIVE
//! node count across sweeps.
//!
//! The per-depth tables above are convergence diagnostics; this is what the
//! driver actually costs. With the adaptive frontier the driver's total node
//! count equals a plain recursive descent's, so these numbers carry over --
//! that was not true of the fixed 8^3 frontier the driver used to build.
static void run_like_driver(
  const char* name, OctBox root, double tol, int cell, double exact)
{
  g_root_volume = root.volume();
  long total = 0;
  int sweeps = 0;
  double secs = 0.0;
  for (int d = 6; d <= 30; ++d) {
    Stats st;
    auto t0 = std::chrono::steady_clock::now();
    descend(root, 0, d, true, st);
    secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
              .count();
    total += st.nodes;
    ++sweeps;
    bool ok = st.lower[cell] <= exact + 1e-12 &&
              exact <= st.lower[cell] + st.slack[cell] + 1e-12;
    if (!ok)
      std::printf("  *** BRACKET VIOLATION at depth %d ***\n", d);
    if (0.5 * st.slack[cell] <= tol) {
      std::printf("  %-22s tol %.0e met at depth %2d after %d sweep(s), "
                  "%ld nodes total, %.3f s  (half-width %.2e)\n",
        name, tol, d, sweeps, total, secs, 0.5 * st.slack[cell]);
      return;
    }
  }
  std::printf(
    "  %-22s tol %.0e NOT met; %ld nodes, %.3f s\n", name, tol, total, secs);
}

int main()
{
  build();
  g_integrators = false;
  double exact = M_PI * R_F * R_F * 2 * HALF_Z;
  std::printf("pincell fuel: analytic volume %.9f cm^3\n", exact);
  std::printf("bounding box %.4f cm^3, fuel volume fraction %.4f\n",
    PITCH * PITCH * 2 * HALF_Z, exact / (PITCH * PITCH * 2 * HALF_Z));
  run(false);
  run(true);

  g_integrators = true;
  std::printf("\n===== with base-case integrators =====\n");
  build();
  run(true);
  run_sphere();

  std::printf("\n===== driver-equivalent cost (mirrors run()) =====\n");
  build();
  run_like_driver("pincell fuel",
    OctBox::from_aabb(
      {-PITCH / 2, -PITCH / 2, -HALF_Z}, {PITCH / 2, PITCH / 2, HALF_Z}),
    1.0e-6, 0, M_PI * R_F * R_F * 2 * HALF_Z);
  return 0;
}
