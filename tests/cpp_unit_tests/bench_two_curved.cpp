// Does subdivision handle two curved surfaces well? Steinmetz solid: the
// intersection of two perpendicular unit cylinders, analytic volume 16 r^3/3.
#include "openmc/volume_octree_math.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <unordered_map>
using namespace openmc;

static vector<QuadricForm> surfs;
static vector<vector<int32_t>> regions;
struct Stats {
  vector<double> lower, slack;
  long nodes {0}, integrated {0}, refused {0};
};

// Minimal integrator: only fires when exactly one cylinder straddles.
static bool try_int(const OctBox& box,
  const std::unordered_map<int32_t, BoxSense>& memo, Stats& st)
{
  vector<int32_t> strad;
  for (auto& kv : memo)
    if (kv.second == BoxSense::BOTH)
      strad.push_back(kv.first);
  std::sort(strad.begin(), strad.end());
  int n = strad.size();
  if (n != 1) {
    if (n > 1)
      ++st.refused;
    return false;
  }
  AxisSymProfile p;
  if (!as_axis_symmetric(surfs[strad[0]], p))
    return false;
  vector<double> px, py;
  double wl, wh;
  if (!box_axis_section(box, p.axis, px, py, wl, wh))
    return false;
  double lo, hi;
  axis_sym_slice_bounds_poly(p, wl, wh, px, py, 4096, lo, hi);
  double vol = box.volume();
  for (int m = 0; m < 2; ++m) {
    auto def = [&](int32_t i) {
      if (i == strad[0])
        return m ? BoxSense::POSITIVE : BoxSense::NEGATIVE;
      auto it = memo.find(i);
      return it == memo.end() ? BoxSense::UNKNOWN : it->second;
    };
    for (int c = 0; c < (int)regions.size(); ++c)
      if (evaluate_region_tri(regions[c], true, def) == Tri::kTrue)
        st.lower[c] += m ? vol - hi : lo;
  }
  for (size_t c = 0; c < regions.size(); ++c)
    st.slack[c] += (hi - lo) + INTEGRATOR_GUARD * vol;
  ++st.integrated;
  return true;
}

static void descend(const OctBox& box, int d, int md, Stats& st)
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
  int nt = 0, it_ = -1;
  bool mb = false;
  for (int c = 0; c < (int)regions.size(); ++c) {
    Tri t = evaluate_region_tri(regions[c], true, sense);
    if (t == Tri::kTrue) {
      ++nt;
      it_ = c;
    } else if (t == Tri::kMaybe)
      mb = true;
  }
  if (nt == 0 && !mb)
    return;
  if (nt == 1 && !mb) {
    st.lower[it_] += box.volume();
    return;
  }
  if (try_int(box, memo, st))
    return;
  if (d >= md) {
    for (int c = 0; c < (int)regions.size(); ++c)
      if (evaluate_region_tri(regions[c], true, sense) != Tri::kFalse)
        st.slack[c] += box.volume();
    return;
  }
  for (int i = 0; i < 8; ++i)
    descend(box.child(i), d + 1, md, st);
}

static QuadricForm cyl(int axis, double r)
{
  QuadricForm q;
  q.A[(axis + 1) % 3] = 1;
  q.A[(axis + 2) % 3] = 1;
  q.c = -r * r;
  return q;
}
static QuadricForm pl(int ax, double v)
{
  QuadricForm q;
  (ax == 0 ? q.b.x : (ax == 1 ? q.b.y : q.b.z)) = 1;
  q.c = -v;
  return q;
}

int main()
{
  double r = 1.0;
  surfs = {cyl(2, r), cyl(0, r)}; // 1: z-cylinder, 2: x-cylinder
  for (int ax = 0; ax < 3; ++ax) {
    surfs.push_back(pl(ax, -2));
    surfs.push_back(pl(ax, 2));
  }
  vector<int32_t> box {3, -4, 5, -6, 7, -8};
  vector<int32_t> in {-1, -2}, out1 {1}, out2 {2};
  for (int32_t t : box) {
    in.push_back(t);
    out1.push_back(t);
    out2.push_back(t);
  }
  regions = {in};
  double exact = 16.0 / 3.0 * r * r * r;
  printf("Steinmetz solid (two perpendicular cylinders), exact %.10f\n", exact);
  printf("%6s %12s %10s %10s %14s %12s\n", "depth", "nodes", "integ", "refused",
    "slack", "half-width");
  double ps = 0;
  long pn = 0;
  for (int d = 1; d <= 11; ++d) {
    Stats st;
    st.lower.assign(regions.size(), 0);
    st.slack.assign(regions.size(), 0);
    auto t0 = std::chrono::steady_clock::now();
    descend(OctBox::from_aabb({-2, -2, -2}, {2, 2, 2}), 0, d, st);
    double sec =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
    bool ok =
      st.lower[0] <= exact + 1e-9 && exact <= st.lower[0] + st.slack[0] + 1e-9;
    printf("%6d %12ld %10ld %10ld %14.6e %12.4e %s", d, st.nodes, st.integrated,
      st.refused, st.slack[0], 0.5 * st.slack[0],
      ok ? "" : "*** VIOLATION ***");
    if (ps > 0)
      printf(
        "  slack x%.2f nodes x%.2f", st.slack[0] / ps, (double)st.nodes / pn);
    printf("  (%.2fs)\n", sec);
    ps = st.slack[0];
    pn = st.nodes;
    if (sec > 25)
      break;
  }
  return 0;
}
