#include "openmc/adjoint_populations.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fmt/core.h>

#include "openmc/bank.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/openmp_interface.h"
#include "openmc/particle.h"
#include "openmc/photonuclear.h"
#include "openmc/physics.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/tally.h"

namespace openmc {

namespace simulation {
bool adjpop_on {false};
} // namespace simulation

namespace adjpop {

namespace {

int n_depth()
{
  return settings::adjpop_n_generation + 1;
}
int n_bins()
{
  return N_SCORE_CLASS * N_TAG * n_depth();
}
size_t bin(int cls, int tag, int depth)
{
  return (static_cast<size_t>(cls) * N_TAG + tag) * n_depth() + depth;
}
int n_ebins()
{
  return settings::adjpop_energy_bins.empty()
           ? 0
           : static_cast<int>(settings::adjpop_energy_bins.size()) - 1;
}
//! Fissioning-nuclide bins: one per listed nuclide and one for the rest;
//! 1 (no split) if none are listed
int n_nbins()
{
  return settings::adjpop_fission_nuclides.empty()
           ? 1
           : static_cast<int>(settings::adjpop_fission_nuclides.size()) + 1;
}
//! Energy groups of the combined tally: 1 (no split) if no edges are given
int n_egroups()
{
  return std::max(n_ebins(), 1);
}
bool tagging_on()
{
  return n_ebins() > 0 || !settings::adjpop_fission_nuclides.empty();
}
//! Combined bins (nuclide bin x energy group), 0 if neither split is on
int n_tbins()
{
  return tagging_on() ? n_nbins() * n_egroups() : 0;
}
int n_ebin_bins()
{
  return n_tbins() * n_depth();
}
int n_lines()
{
  return static_cast<int>(settings::adjpop_probe_energies.size());
}
bool probes_on()
{
  return n_lines() > 0;
}
//! Probe labels: 0 uncollided, 1 after coherent scattering only, 2 after an
//! energy-changing collision (or a secondary photon)
constexpr int N_PROBE_LABEL = 3;
//! Probe bins: nuclide bin x line x label
int n_probe_bins()
{
  return n_nbins() * n_lines() * N_PROBE_LABEL;
}
int n_probe_depth_bins()
{
  return n_probe_bins() * n_depth();
}
// The rays' photon lines (the probe lines subdivided) and their neutron comb,
// set by init()
vector<double> ray_lines;
vector<double> ray_comb;
int n_ray_lines()
{
  return static_cast<int>(ray_lines.size());
}
int n_comb()
{
  return static_cast<int>(ray_comb.size());
}
bool rays_on()
{
  return probes_on() && n_comb() > 1;
}
bool rays_requested()
{
  return !settings::adjpop_ray_neutron_energies.empty() ||
         settings::adjpop_ray_comb_points > 0;
}
//! Ray scores: per line (nuclide bin x line) and per comb energy (nuclide
//! bin x comb energy), each by depth
int n_ray_line_bins()
{
  return rays_on() ? n_nbins() * n_ray_lines() * n_depth() : 0;
}
int n_ray_comb_bins()
{
  return rays_on() ? n_nbins() * n_comb() * n_depth() : 0;
}

// Bin of each nuclide in data::nuclides (the "rest" bin if not listed)
vector<int> nuclide_bin;

// Roots and fission events recorded by the driver this generation
vector<vector<Root>> thread_roots;
vector<vector<FissionRecord>> thread_fissions;

// Target (site) weight of each population and branch class for the current
// shadow pass. A branch target of 0 means not yet set.
double site_w[N_SCORE_CLASS] {1.0, 1.0, 1.0, 0.0, 0.0};

// Per-thread raw weight and count of branch photoneutrons (before and after
// their roulette) in the current shadow pass, per branched class
vector<double> thread_branch_raw;
vector<int64_t> thread_branch_n;

// Per-thread scoring slabs: summed weight, and weight times lifetime stamp
vector<double> thread_w;
vector<double> thread_wt;

// Photoneutron-root weight per energy group and depth: per thread, this
// batch, every finished batch
vector<double> thread_ew;
vector<double> batch_ew;
vector<double> batches_ew;

// Probe-root weight per probe bin and depth, and the probed fission weight
// (sum of w/k sigma_f/sigma_t) per nuclide bin: per thread, this batch, every
// finished batch
vector<double> thread_pw;
vector<double> batch_pw;
vector<double> batches_pw;
vector<double> batch_pf;
vector<double> batches_pf;

// Probe roots recorded by the probe photons of this shadow pass
vector<vector<Root>> thread_probe_roots;

// Ray probes: per material and line, the total photon cross section and the
// photoneutron channels (photonuclear nuclide, reaction, product, and its
// production N sigma y); their scores per line and per comb energy
struct Channel {
  int i_pn;
  int i_rx;
  int i_prod;
  double share;
};
vector<double> ray_sigt;              // [material][line]
vector<double> ray_sign;              // [material][line]
vector<double> ray_sigt_min;          // [material], smallest over lines
vector<vector<Channel>> ray_channels; // [material][line]
vector<double> thread_rw, batch_rw, batches_rw;
vector<double> thread_rj, batch_rj, batches_rj;
vector<double> batch_rf, batches_rf;
// Per fissioning-nuclide bin: the ray roulette targets (0: keep all) and the
// site weight the bin's ray trees bank at
vector<double> ray_cast_target;
vector<double> ray_root_target;
vector<double> ray_site_w;
int64_t last_rays {0};
int64_t last_ray_roots {0};
int64_t n_ray_segments {0};
// The depth scores of the tree being grown on this thread, when it is a ray
// tree (shared by every line with its own factor)
vector<vector<double>> thread_tree_buf;
vector<char> thread_tree_active;

// The probe photons' targets per fissioning-nuclide bin; the probe roots'
// targets and the site weight their trees bank at per bin and line
vector<double> probe_photon_target;
vector<double> probe_root_target;
vector<double> probe_site_w;

// This batch's sums, and every finished batch's
vector<double> batch_w;
vector<double> batch_wt;
vector<double> batches_w;
vector<double> batches_wt;
int n_batches_recorded {0};

// Diagnostics of the last shadow pass
int64_t last_roots[N_SCORE_CLASS] {0, 0, 0, 0, 0};
double last_raw_weight[N_SCORE_CLASS] {0.0, 0.0, 0.0, 0.0, 0.0};
int64_t n_histories {0};
int64_t last_probe_photons {0};
int64_t last_probe_roots {0};
int64_t n_probe_histories {0};

int64_t generation_key()
{
  // Particle ids restart every generation and total_gen only advances at
  // finalize, so the generation must be keyed explicitly or every seed
  // repeats from one generation to the next.
  return simulation::total_gen + overall_generation();
}

int64_t bits(double x)
{
  int64_t b;
  std::memcpy(&b, &x, sizeof(b));
  return b;
}

void score(int cls, int tag, int depth, double w, double t0)
{
  const size_t slab = static_cast<size_t>(thread_num()) * n_bins();
  const size_t i = slab + bin(cls, tag, depth);
  thread_w[i] += w;
  thread_wt[i] += w * t0;
}

void score_ebin(int ebin, int depth, double w)
{
  if (ebin < 0)
    return;
  const size_t i = static_cast<size_t>(thread_num()) * n_ebin_bins() +
                   static_cast<size_t>(ebin) * n_depth() + depth;
  thread_ew[i] += w;
}

//! A probe tree's score: its bin is (nuclide bin x line) in the ebin field of
//! the tag and the scattered flag in its tag field
void score_probe(int ebin, int flag, int depth, double w)
{
  const size_t i = static_cast<size_t>(thread_num()) * n_probe_depth_bins() +
                   (static_cast<size_t>(ebin) * N_PROBE_LABEL + flag) * n_depth() +
                   depth;
  thread_pw[i] += w;
}

//! Every score of a tree at one depth, by its class
void score_tree(int cls, int tag, int ebin, int depth, double w, double t0)
{
  if (cls == CLASS_RAY) {
    thread_tree_buf[thread_num()][depth] += w;
  } else if (cls == CLASS_PROBE) {
    score_probe(ebin, tag, depth, w);
  } else {
    score(cls, tag, depth, w, t0);
    score_ebin(ebin, depth, w);
  }
}

//! Nuclide bin of a nuclide in data::nuclides
int nuclide_bin_of(int i_nuclide)
{
  if (settings::adjpop_fission_nuclides.empty())
    return 0;
  return (i_nuclide >= 0 && i_nuclide < static_cast<int>(nuclide_bin.size()))
           ? nuclide_bin[i_nuclide]
           : n_nbins() - 1;
}

//! Target (site) weight of a class
double target_weight(int cls, int ebin = -1)
{
  if (cls == CLASS_RAY)
    return ray_site_w[ebin / std::max(n_comb(), 1)];
  if (cls == CLASS_PROBE)
    return probe_site_w[ebin];
  return site_w[cls];
}

//! Combined bin of a photoneutron, nuclide bin x n_egroups + energy group:
//! the nuclide whose fission made the photon, and the photon's birth energy
//! or the photoneutron's own. -1 if tagging is off or the energy is outside
//! the edges. With the nuclide split off it is the energy group alone.
int energy_group(const Particle& photon, double E_neutron)
{
  if (!tagging_on())
    return -1;
  int group = 0;
  const auto& e = settings::adjpop_energy_bins;
  if (!e.empty()) {
    const double x =
      (settings::adjpop_energy_variable == 0) ? photon.E_born() : E_neutron;
    if (x < e.front() || x >= e.back())
      return -1;
    group =
      static_cast<int>(std::upper_bound(e.begin(), e.end(), x) - e.begin()) - 1;
  }
  return nuclide_bin_of(photon.fission_nuclide()) * n_egroups() + group;
}

//! Russian roulette of a root to its population's target weight, drawn on
//! its own key so that no tree replays the number deciding its existence.
//! \return the weight the root is grown with, or 0 if it is killed
double roulette_to(double w_raw, double target, int64_t seed_id)
{
  if (!(w_raw > 0.0))
    return 0.0;
  if (w_raw >= target)
    return w_raw;
  uint64_t s = init_seed(combine_ids({seed_id, 1}), STREAM_TRACKING);
  return (prn(&s) < w_raw / target) ? target : 0.0;
}

double roulette(double w_raw, int cls, int64_t seed_id)
{
  return roulette_to(w_raw, target_weight(cls), seed_id);
}

//! Split or roulette a weight to copies of the target weight: floor(w/t)
//! copies and one more with probability the remainder, so that the expected
//! weight is kept. A target of 0 keeps one copy of the raw weight. The draw
//! is roulette_to()'s, so a weight below the target meets the same fate.
//! \return the number of copies; w_copy is set to their weight
int split_to(double w_raw, double target, int64_t seed_id, double& w_copy)
{
  w_copy = 0.0;
  if (!(w_raw > 0.0))
    return 0;
  if (!(target > 0.0)) {
    w_copy = w_raw;
    return 1;
  }
  const double x = std::min(w_raw / target, static_cast<double>(MAX_SPLIT));
  int n = static_cast<int>(x);
  uint64_t s = init_seed(combine_ids({seed_id, 1}), STREAM_TRACKING);
  if (prn(&s) < x - n)
    ++n;
  // Beyond MAX_SPLIT copies the copies share the whole weight
  w_copy = (w_raw / target > MAX_SPLIT) ? w_raw / MAX_SPLIT : target;
  return n;
}

//! The seed key of copy c of a split event (copy 0 keeps the event's own)
int64_t copy_id(int64_t id, int c)
{
  return (c == 0) ? id : combine_ids({id, c, 808});
}

//! Grow one root n_generation generations in the unperturbed physics
void run_one_tree(const Root& root, double w)
{
  SourceSite site;
  site.r = root.r;
  site.u = root.u;
  site.E = root.E;
  site.time = root.time;
  site.wgt = w;
  site.particle = ParticleType::neutron();
  site.shadow_depth = 0;
  site.shadow_tag = pack_tag(root.cls, root.tag, root.ebin);
  site.shadow_t0 = 0.0;
  site.wgt_born = w;

  Particle p;
  p.from_source(&site);
  p.id() = root.seed_id;
  p.n_progeny() = 0;
  p.n_event() = 0;
  p.n_tracks() = 1;
  p.n_split() = 0;
  p.write_track() = false;
  p.current_work() = 1;
  // The tree grows on its own stream, keyed differently from the one that
  // sampled the root's outgoing state.
  init_particle_seeds(combine_ids({root.seed_id, 2}), p.seeds());
  p.stream() = STREAM_TRACKING;

  score_tree(root.cls, root.tag, root.ebin, 0, w, 0.0);
  transport_history_based_single_particle(p);
#pragma omp atomic
  n_histories += p.n_tracks();
}

//! A probe photon waiting to be transported
struct Probe {
  Position r;
  double time;
  double wgt; //!< weight after the roulette
  int line;
  int nbin;
  int64_t seed_id;
};

//! The probe photon of one driver fission event. Decided first (roulette on
//! its own key), sampled second, as for the delayed roots.
void emit_probe(const FissionRecord& f, vector<Probe>& out)
{
  const double w_raw = f.wgt * f.sigma_f_to_t * n_lines();
  const int64_t sid = combine_ids({f.seed_id, 404});
  const int nbin = nuclide_bin_of(f.i_nuclide);
  double w;
  const int n = split_to(w_raw, probe_photon_target[nbin], sid, w);
  if (n == 0)
    return;
  // The copies' lines are stratified: each is uniform over the lines, and
  // together they spread over the comb
  uint64_t s = init_seed(combine_ids({sid, 3}), STREAM_TRACKING);
  const double u = prn(&s);
  for (int c = 0; c < n; ++c) {
    Probe q;
    q.r = f.r;
    q.time = f.time;
    q.wgt = w;
    q.line = std::min(
      static_cast<int>((u + c) / n * n_lines()), n_lines() - 1);
    q.nbin = nbin;
    q.seed_id = copy_id(sid, c);
    out.push_back(q);
  }
}

//! Transport one probe photon; its photoneutrons are recorded as probe roots
//! by record_probe_photoneutron()
void run_one_probe(const Probe& q)
{
  uint64_t s = init_seed(combine_ids({q.seed_id, 5}), STREAM_TRACKING);
  const double cos_theta = 2.0 * prn(&s) - 1.0;
  const double phi = 2.0 * PI * prn(&s);
  const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

  SourceSite site;
  site.r = q.r;
  site.u = {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
  site.E = settings::adjpop_probe_energies[q.line];
  site.time = q.time;
  site.wgt = q.wgt;
  site.particle = ParticleType::photon();
  site.shadow_depth = 0;
  site.shadow_tag = pack_tag(CLASS_PROBE, 0, q.nbin * n_lines() + q.line);
  site.shadow_t0 = 0.0;
  site.wgt_born = q.wgt;

  Particle p;
  p.from_source(&site);
  p.id() = q.seed_id;
  p.n_progeny() = 0;
  p.n_event() = 0;
  p.n_tracks() = 1;
  p.n_split() = 0;
  p.write_track() = false;
  p.current_work() = 1;
  init_particle_seeds(combine_ids({q.seed_id, 2}), p.seeds());
  p.stream() = STREAM_TRACKING;
  transport_history_based_single_particle(p);
#pragma omp atomic
  n_probe_histories += p.n_tracks();
}

//! Delayed roots of one driver fission event, one per group
void emit_delayed(const FissionRecord& f, vector<std::pair<Root, double>>& out,
  double& raw_sum, bool keep)
{
  const auto& nuc = *data::nuclides[f.i_nuclide];
  const Reaction& rx = *f.rx;
  const int n_group =
    std::min<int>(nuc.n_precursor_, static_cast<int>(rx.products_.size()) - 1);
  for (int g = 1; g <= n_group; ++g) {
    const double w_raw =
      f.wgt * f.sigma_f_to_t * (*rx.products_[g].yield_)(f.E);
    raw_sum += w_raw;
    if (!keep)
      continue;
    const int64_t sid = combine_ids({f.seed_id, g});
    const double w = roulette(w_raw, CLASS_DELAYED, sid);
    if (w == 0.0)
      continue;

    // Decided first, sampled second: a killed root is never sampled.
    uint64_t s = init_seed(combine_ids({sid, 3}), STREAM_TRACKING);
    Root r;
    r.r = f.r;
    r.time = f.time;
    r.wgt = w_raw;
    r.cls = CLASS_DELAYED;
    r.tag = std::min(g, N_TAG - 1);
    r.seed_id = sid;
    const int neutron = ParticleType::neutron().transport_index();
    double mu;
    int n_sample = 0;
    while (true) {
      rx.products_[g].sample(f.E, r.E, mu, &s);
      if (r.E < data::energy_max[neutron])
        break;
      if (++n_sample == MAX_SAMPLE)
        fatal_error("Resampled a delayed-neutron spectrum the maximum number "
                    "of times for nuclide " +
                    nuc.name_);
    }
    // Delayed neutrons are emitted isotropically
    const double cos_theta = 2.0 * prn(&s) - 1.0;
    const double phi = 2.0 * PI * prn(&s);
    const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    r.u = {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
    out.emplace_back(r, w);
  }
}

//! A ray probe's photoneutron root: the root, the nuclide bin, and every
//! line's factor on the tree's score (divided by the root's raw weight)
struct RayRoot {
  Root root;
  int nbin;
  vector<double> factor;
};

//! One straight segment of a ray: material, length, start and direction
struct RaySeg {
  int mat;
  double len;
  Position r;
  Direction u;
};

//! Straight-line walk of an uncollided photon through the geometry,
//! recording (material, length) of every segment. Surfaces are crossed as
//! in transport, so reflective and periodic boundaries continue the walk and
//! a vacuum boundary ends it; so does an optical depth beyond any line's
//! contribution.
void walk_ray(Position r0, Direction u, uint64_t* seed, vector<RaySeg>& seg)
{
  Particle p;
  p.type() = ParticleType::photon();
  p.wgt() = 1.0;
  p.E() = ray_lines.back();
  p.shadow_depth() = 0;
  p.r() = r0;
  p.u() = u;
  p.r_last() = r0;
  p.u_last() = u;
  // White boundaries sample a direction
  p.seeds(0) = *seed;
  p.stream() = 0;
  if (!exhaustive_find_cell(p, false))
    return;
  double tau_min = 0.0;
  for (int n = 0; n < 1000000; ++n) {
    const int mat = p.material();
    p.boundary() = distance_to_boundary(p);
    const double d = p.boundary().distance();
    if (!(d < INFTY) || d < 0.0)
      return;
    seg.push_back({mat, d, p.r(), p.u()});
    if (mat >= 0)
      tau_min += ray_sigt_min[mat] * d;
    if (tau_min > 50.0)
      return;
    for (int lev = 0; lev < p.n_coord(); ++lev)
      p.coord(lev).r() += d * p.coord(lev).u();
    p.event_cross_surface();
    if (p.wgt() == 0.0 || p.lowest_coord().cell() == C_NONE)
      return;
  }
}

//! Build the per-material, per-line cross sections of the ray probes
void init_rays()
{
  const int K = n_ray_lines();
  const size_t nm = model::materials.size();
  ray_sigt.assign(nm * K, 0.0);
  ray_sign.assign(nm * K, 0.0);
  ray_sigt_min.assign(nm, 0.0);
  ray_channels.assign(nm * K, {});
  Particle p;
  p.type() = ParticleType::photon();
  for (size_t m = 0; m < nm; ++m) {
    const auto& mat = *model::materials[m];
    double tmin = INFTY;
    for (int k = 0; k < K; ++k) {
      p.E() = ray_lines[k];
      mat.calculate_xs(p);
      ray_sigt[m * K + k] = p.macro_xs().total;
      tmin = std::min(tmin, p.macro_xs().total);
      double sn = 0.0;
      for (int i = 0; i < static_cast<int>(mat.nuclide_.size()); ++i) {
        const auto& name = data::nuclides[mat.nuclide_[i]]->name_;
        auto it = data::photonuclear_map.find(name);
        if (it == data::photonuclear_map.end())
          continue;
        const int i_pn = it->second;
        const auto& micro = p.photonuclear_xs(i_pn);
        const auto& nuc = *data::photonuclears[i_pn];
        // As emit_forced_photoneutron() samples them: every reaction's cross
        // section once per neutron product, times that product's yield
        for (int r = 0; r < static_cast<int>(nuc.reactions_.size()); ++r) {
          const auto& rx = *nuc.reactions_[r];
          const double xs = rx.xs(micro);
          if (!(xs > 0.0))
            continue;
          for (int q = 0; q < static_cast<int>(rx.products_.size()); ++q) {
            if (rx.products_[q].particle_ != ParticleType::neutron())
              continue;
            const double y = (*rx.products_[q].yield_)(p.E());
            if (!(y > 0.0))
              continue;
            const double share = mat.atom_density_(i) * xs * y;
            ray_channels[m * K + k].push_back({i_pn, r, q, share});
            sn += share;
          }
        }
      }
      ray_sign[m * K + k] = sn;
    }
    ray_sigt_min[m] = (tmin < INFTY) ? tmin : 0.0;
  }
}

//! The neutron comb when it is not given: n energies at equally spaced
//! quantiles (0.1 % to 99.9 %) of the laboratory photoneutron energies the
//! rays make, so that every comb energy gets about the same share of the
//! trees. The lines are weighted as the ray's comb draw weights them, half
//! by their photoneutron production and half equally; each line's
//! photoneutrons are sampled over its channels (a centre-of-mass channel
//! isotropic there and moved to the laboratory). Not below 1 eV.
void build_ray_comb(int n)
{
  const int K = n_ray_lines();
  vector<std::pair<double, double>> sample; // (lethargy, weight)
  uint64_t s = init_seed(combine_ids({4242, 17}), STREAM_TRACKING);
  constexpr int N_SAMPLE = 256;
  for (size_t m = 0; m < model::materials.size(); ++m) {
    double sn_sum = 0.0;
    int live = 0;
    for (int k = 0; k < K; ++k) {
      sn_sum += ray_sign[m * K + k];
      live += ray_sign[m * K + k] > 0.0;
    }
    if (!(sn_sum > 0.0))
      continue;
    for (int k = 0; k < K; ++k) {
      const double sn = ray_sign[m * K + k];
      if (!(sn > 0.0))
        continue;
      const double w_line = 0.5 * sn / sn_sum + 0.5 / live;
      const double Ek = ray_lines[k];
      const auto& ch = ray_channels[m * K + k];
      for (int i = 0; i < N_SAMPLE; ++i) {
        // A channel in proportion to its share
        double xi = prn(&s) * sn;
        size_t c = 0;
        for (; c + 1 < ch.size(); ++c) {
          if (xi < ch[c].share)
            break;
          xi -= ch[c].share;
        }
        const auto& nuc = *data::photonuclears[ch[c].i_pn];
        const auto& rx = *nuc.reactions_[ch[c].i_rx];
        double E_out;
        rx.products_[ch[c].i_prod].sample_energy_and_pdf(
          Ek, 2.0 * prn(&s) - 1.0, E_out, &s);
        if (!(E_out > 0.0))
          continue;
        double E_lab = E_out;
        if (rx.scatter_in_cm_) {
          const double a = std::sqrt(E_out);
          const double b = Ek / (nuc.awr_ * std::sqrt(2.0 * MASS_NEUTRON_EV));
          E_lab = a * a + b * b + 2.0 * a * b * (2.0 * prn(&s) - 1.0);
        }
        sample.emplace_back(
          std::log(std::max(E_lab, 1.0)), w_line / N_SAMPLE);
      }
    }
  }
  if (sample.empty())
    fatal_error("<adjoint_populations> no photoneutron channel is open at "
                "the ray lines: the neutron comb cannot be built.");
  std::sort(sample.begin(), sample.end());
  double total = 0.0;
  for (const auto& x : sample)
    total += x.second;
  ray_comb.assign(n, 0.0);
  double cum = 0.0;
  size_t i = 0;
  for (int j = 0; j < n; ++j) {
    const double q = (0.001 + 0.998 * j / (n - 1)) * total;
    while (i + 1 < sample.size() && cum + sample[i].second < q) {
      cum += sample[i].second;
      ++i;
    }
    ray_comb[j] = std::exp(sample[i].first);
  }
  // Strictly increasing
  for (int j = 1; j < n; ++j)
    ray_comb[j] = std::max(ray_comb[j], ray_comb[j - 1] * 1.001);
}

//! Linear-interpolation weights in lethargy of a photoneutron energy on the
//! comb, held at the ends
void comb_weights(double E, int& j0, double& w0)
{
  const auto& c = ray_comb;
  const int M = static_cast<int>(c.size());
  if (E <= c.front()) {
    j0 = 0;
    w0 = 1.0;
    return;
  }
  if (E >= c.back()) {
    j0 = M - 2;
    w0 = 0.0;
    return;
  }
  j0 = static_cast<int>(std::upper_bound(c.begin(), c.end(), E) - c.begin()) - 1;
  w0 = std::log(c[j0 + 1] / E) / std::log(c[j0 + 1] / c[j0]);
}

//! Laboratory angular density of one photoneutron channel at the laboratory
//! cosine mu (about the photon's direction), and the photoneutron's laboratory
//! energy, for a photon of energy E_in. A distribution given in the
//! laboratory is evaluated as it is. One given in the centre of mass is
//! transformed: its outgoing energy is drawn, the two-body kinematics solved
//! for the centre-of-mass cosine (two roots when the centre of mass outruns
//! the neutron), and the density carries the Jacobian. The energy draw is
//! replayed from the same seed when the density is evaluated, so energy and
//! angle belong to one sample; the result is an unbiased estimate of the
//! laboratory density (exact for two-body reactions). Returns the number of
//! roots, each with its density contribution and laboratory energy.
int lab_density(const ReactionProduct& prod, bool in_cm, double E_in,
  double awr, double mu, uint64_t* seed, double dens[2], double E_lab[2])
{
  if (!in_cm) {
    double E_out;
    dens[0] = prod.sample_energy_and_pdf(E_in, mu, E_out, seed);
    E_lab[0] = E_out;
    return 1;
  }
  const uint64_t s0 = *seed;
  uint64_t s1 = s0;
  double E_cm;
  prod.sample_energy_and_pdf(E_in, 0.0, E_cm, &s1);
  *seed = s1;
  if (!(E_cm > 0.0))
    return 0;
  const double a = std::sqrt(E_cm);
  const double b = E_in / (awr * std::sqrt(2.0 * MASS_NEUTRON_EV));
  const double D2 = a * a - b * b * (1.0 - mu * mu);
  if (D2 <= 0.0)
    return 0;
  const double D = std::sqrt(D2);
  int n = 0;
  for (double sg : {1.0, -1.0}) {
    const double sp = b * mu + sg * D;
    if (!(sp > 0.0))
      continue;
    const double mu_cm = std::clamp((sp * mu - b) / a, -1.0, 1.0);
    uint64_t s2 = s0;
    double E_cm2;
    const double pdf = prod.sample_energy_and_pdf(E_in, mu_cm, E_cm2, &s2);
    dens[n] = pdf * sp * sp / (a * D);
    E_lab[n] = sp * sp;
    ++n;
  }
  return n;
}

//! Cast the ray of one driver fission event and turn it into one shared
//! photoneutron root. Returns false if the ray makes no photoneutron.
bool cast_ray(const FissionRecord& f, double w_ray, int64_t sid, RayRoot& out,
  int64_t& n_seg)
{
  const int K = n_ray_lines();
  const int M = n_comb();
  uint64_t s = init_seed(combine_ids({sid, 3}), STREAM_TRACKING);
  const double ct = 2.0 * prn(&s) - 1.0;
  const double ph = 2.0 * PI * prn(&s);
  const double st = std::sqrt(1.0 - ct * ct);
  const Direction u {st * std::cos(ph), st * std::sin(ph), ct};

  vector<RaySeg> seg;
  walk_ray(f.r, u, &s, seg);
  n_seg += static_cast<int64_t>(seg.size());
  const int ns = static_cast<int>(seg.size());
  if (ns == 0)
    return false;

  // Optical depth at each segment's start and each line's production
  vector<double> tau((ns + 1) * K, 0.0);
  vector<double> cum((ns + 1) * K, 0.0);
  for (int i = 0; i < ns; ++i) {
    const int m = seg[i].mat;
    const double L = seg[i].len;
    for (int k = 0; k < K; ++k) {
      const double t0 = tau[i * K + k];
      double st_ = 0.0, sn = 0.0;
      if (m >= 0) {
        st_ = ray_sigt[m * K + k];
        sn = ray_sign[m * K + k];
      }
      tau[(i + 1) * K + k] = t0 + st_ * L;
      const double add = (st_ > 0.0 && sn > 0.0)
                           ? sn / st_ * std::exp(-t0) * (-std::expm1(-st_ * L))
                           : 0.0;
      cum[(i + 1) * K + k] = cum[i * K + k] + add;
    }
  }
  vector<double> Y(K);
  double Ysum = 0.0;
  for (int k = 0; k < K; ++k) {
    Y[k] = cum[ns * K + k];
    Ysum += Y[k];
  }
  if (!(Ysum > 0.0))
    return false;

  // Birth point: a line in proportion to its production, then a point from
  // that line's production density along the ray
  double xi = prn(&s) * Ysum;
  int ks = 0;
  for (; ks < K - 1; ++ks) {
    if (xi < Y[ks])
      break;
    xi -= Y[ks];
  }
  const double target_c = prn(&s) * Y[ks];
  int ib = 0;
  while (ib < ns - 1 && cum[(ib + 1) * K + ks] < target_c)
    ++ib;
  const int mb = seg[ib].mat;
  const double stb = ray_sigt[mb * K + ks];
  const double snb = ray_sign[mb * K + ks];
  // Within the segment: e^{-tau} falls by (c - cum_start) st/sn
  const double e0 = std::exp(-tau[ib * K + ks]);
  double e = e0 - (target_c - cum[ib * K + ks]) * stb / snb;
  e = std::max(e, 1e-300);
  double ds = (-std::log(e) - tau[ib * K + ks]) / stb;
  ds = std::clamp(ds, 0.0, seg[ib].len);
  const Position rb = seg[ib].r + ds * seg[ib].u;
  const Direction ub = seg[ib].u;

  // Every line's production density there
  vector<double> pk(K);
  double psum = 0.0;
  for (int k = 0; k < K; ++k) {
    const double sn = ray_sign[mb * K + k];
    const double tk = tau[ib * K + k] + ray_sigt[mb * K + k] * ds;
    pk[k] = sn * std::exp(-tk);
    psum += pk[k];
  }
  if (!(psum > 0.0))
    return false;

  // Direction: isotropic in the laboratory about the photon, a density known
  // exactly (1/2 in the cosine) whatever the reactions' distributions
  const double mu = 2.0 * prn(&s) - 1.0;
  const Direction un = rotate_angle(ub, mu, nullptr, &s);

  // Per line: the angular density at mu and the comb weights of its
  // photoneutron energy, over its channels
  vector<double> fk(K, 0.0);
  vector<double> wc(static_cast<size_t>(K) * M, 0.0);
  for (int k = 0; k < K; ++k) {
    if (!(pk[k] > 0.0))
      continue;
    const double Ek = ray_lines[k];
    double fsum = 0.0, ssum = 0.0;
    for (const auto& c : ray_channels[mb * K + k]) {
      const auto& nuc = *data::photonuclears[c.i_pn];
      const auto& rx = *nuc.reactions_[c.i_rx];
      double dens[2], E_lab[2];
      const int nr = lab_density(rx.products_[c.i_prod], rx.scatter_in_cm_, Ek,
        nuc.awr_, mu, &s, dens, E_lab);
      ssum += c.share;
      for (int q = 0; q < nr; ++q) {
        if (!(dens[q] > 0.0))
          continue;
        fsum += c.share * dens[q];
        int j0;
        double w0;
        comb_weights(E_lab[q], j0, w0);
        wc[static_cast<size_t>(k) * M + j0] += c.share * dens[q] * w0;
        wc[static_cast<size_t>(k) * M + j0 + 1] +=
          c.share * dens[q] * (1.0 - w0);
      }
    }
    if (!(ssum > 0.0) || !(fsum > 0.0))
      continue;
    for (int j = 0; j < M; ++j)
      wc[static_cast<size_t>(k) * M + j] /= fsum;
    fk[k] = fsum / ssum;
  }

  // Line weights: the birth point was drawn from the lines' mixture,
  // Ysum^-1 sum_k p_k(s), and the cosine from 1/2, so line k's weight is its
  // density p_k(s) f_k(mu) over theirs. Then one comb energy for all lines,
  // drawn from an even mixture of the lines' comb weights taken by their
  // share of the ray and taken equally, so that a line that makes few
  // photoneutrons still gets trees of its own. Each line's factor divides by
  // the probability of the energy drawn, so every line stays unbiased.
  vector<double> b(K, 0.0);
  double bsum = 0.0;
  int n_live = 0;
  for (int k = 0; k < K; ++k) {
    b[k] = w_ray * Ysum * pk[k] * fk[k] * 2.0 / psum;
    bsum += b[k];
    if (b[k] > 0.0)
      ++n_live;
  }
  if (!(bsum > 0.0))
    return false;
  vector<double> q(M, 0.0);
  for (int k = 0; k < K; ++k) {
    if (!(b[k] > 0.0))
      continue;
    const double share = 0.5 * b[k] / bsum + 0.5 / n_live;
    for (int j = 0; j < M; ++j)
      q[j] += share * wc[static_cast<size_t>(k) * M + j];
  }
  double qs = 0.0;
  for (int j = 0; j < M; ++j)
    qs += q[j];
  if (!(qs > 0.0))
    return false;
  double xj = prn(&s) * qs;
  int jc = 0;
  for (; jc < M - 1; ++jc) {
    if (xj < q[jc])
      break;
    xj -= q[jc];
  }
  const double qj = q[jc] / qs;

  out.factor.assign(K, 0.0);
  double R = 0.0;
  for (int k = 0; k < K; ++k) {
    out.factor[k] = b[k] * wc[static_cast<size_t>(k) * M + jc] / qj;
    R += out.factor[k];
  }
  R /= K;
  if (!(R > 0.0))
    return false;
  for (auto& a : out.factor)
    a /= R;
  out.nbin = nuclide_bin_of(f.i_nuclide);
  Root& r = out.root;
  r.r = rb;
  r.u = un;
  r.E = ray_comb[jc];
  r.time = f.time;
  r.wgt = R;
  r.cls = CLASS_RAY;
  r.tag = 0;
  r.ebin = out.nbin * M + jc;
  r.seed_id = combine_ids({sid, 606});
  return true;
}

//! A population's root fraction: its own if set (positive), else the common
double root_fraction_or_default(double f)
{
  return (f > 0.0) ? f : settings::adjpop_root_fraction;
}

//! Share a budget of m roots among nuclide bins with raw weights and counts:
//! equally, each bin capped at its count (then it keeps all, target 0), the
//! rest to the others; or in proportion to the weights. Returns the targets.
vector<double> share_targets(
  const vector<double>& raw, const vector<int64_t>& count, double m)
{
  const int nb = static_cast<int>(raw.size());
  vector<double> target(nb, 0.0);
  double raw_sum = 0.0;
  for (double r : raw)
    raw_sum += r;
  if (settings::adjpop_ray_allocation == 1) {
    for (int b = 0; b < nb; ++b)
      target[b] = raw_sum / m;
    return target;
  }
  vector<char> left(nb, 0);
  int n_left = 0;
  for (int b = 0; b < nb; ++b)
    if (raw[b] > 0.0 && count[b] > 0) {
      left[b] = 1;
      ++n_left;
    }
  double remaining = m;
  bool changed = true;
  while (changed && n_left > 0) {
    changed = false;
    const double share = remaining / n_left;
    for (int b = 0; b < nb; ++b) {
      if (left[b] && static_cast<double>(count[b]) <= share) {
        target[b] = 0.0;
        remaining -= static_cast<double>(count[b]);
        left[b] = 0;
        --n_left;
        changed = true;
      }
    }
  }
  if (n_left > 0) {
    const double share = std::max(remaining, 1.0) / n_left;
    for (int b = 0; b < nb; ++b)
      if (left[b])
        target[b] = raw[b] / share;
  }
  return target;
}

//! share_targets() for events that may be split: a bin holds up to MAX_SPLIT
//! copies of each of its events, and a bin that gets all it can hold splits
//! every event MAX_SPLIT-fold
vector<double> split_targets(
  const vector<double>& raw, const vector<int64_t>& count, double m)
{
  vector<int64_t> cap(count.size());
  for (size_t b = 0; b < count.size(); ++b)
    cap[b] = count[b] * MAX_SPLIT;
  vector<double> target = share_targets(raw, cap, m);
  for (size_t b = 0; b < count.size(); ++b)
    if (target[b] == 0.0 && cap[b] > 0)
      target[b] = raw[b] / static_cast<double>(cap[b]);
  return target;
}

//! Grow a ray root and share its depth scores among the lines
void run_one_ray_tree(const RayRoot& rr, double w)
{
  const int t = thread_num();
  auto& buf = thread_tree_buf[t];
  std::fill(buf.begin(), buf.end(), 0.0);
  run_one_tree(rr.root, w);
  const int K = n_ray_lines();
  const size_t nd = n_depth();
  const size_t lines = static_cast<size_t>(t) * n_ray_line_bins();
  const size_t comb = static_cast<size_t>(t) * n_ray_comb_bins();
  for (int k = 0; k < K; ++k) {
    const double a = rr.factor[k];
    if (a == 0.0)
      continue;
    const size_t base = lines + (static_cast<size_t>(rr.nbin) * K + k) * nd;
    for (size_t d = 0; d < nd; ++d)
      thread_rw[base + d] += buf[d] * a;
  }
  const size_t cb = comb + static_cast<size_t>(rr.root.ebin) * nd;
  for (size_t d = 0; d < nd; ++d)
    thread_rj[cb + d] += buf[d];
}

} // namespace

//==============================================================================

void init()
{
  if (settings::adjpop_n_generation <= 0)
    return;
  if (settings::run_mode != RunMode::EIGENVALUE)
    fatal_error("<adjoint_populations> requires an eigenvalue calculation.");
  if (!settings::run_CE)
    fatal_error("<adjoint_populations> requires continuous-energy data.");
  if (settings::event_based)
    fatal_error("<adjoint_populations> requires history-based transport.");
  if (settings::use_shared_secondary_bank)
    fatal_error("<adjoint_populations> is not compatible with the shared "
                "secondary bank: shadow-tree sites are revived from the "
                "particle's own secondary bank.");
  if (settings::weight_windows_on)
    fatal_error("<adjoint_populations> is not compatible with weight windows: "
                "they would act on shadow-tree particles, whose weights are "
                "set by their populations' target weights.");
  if (settings::survival_biasing)
    fatal_error("<adjoint_populations> is not compatible with survival "
                "biasing.");
  if (settings::ufs_on)
    fatal_error("<adjoint_populations> is not compatible with uniform fission "
                "site weighting.");
  if (settings::adjpop_photoneutrons && !settings::photonuclear_physics)
    fatal_error("<adjoint_populations> photoneutrons require "
                "<photonuclear_physics> to be on.");
  if (settings::adjpop_perturbed_importance && !settings::adjpop_photoneutrons)
    fatal_error("<adjoint_populations> perturbed_importance requires "
                "photoneutrons to be on.");

  const int n_threads = num_threads();
  thread_roots.assign(n_threads, {});
  thread_fissions.assign(n_threads, {});
  thread_w.assign(static_cast<size_t>(n_threads) * n_bins(), 0.0);
  thread_wt.assign(static_cast<size_t>(n_threads) * n_bins(), 0.0);
  if (tagging_on() && !settings::adjpop_photoneutrons)
    fatal_error("<adjoint_populations> photoneutron_energy_bins and "
                "photoneutron_fission_nuclides require photoneutrons to be on.");
  nuclide_bin.assign(data::nuclides.size(), n_nbins() - 1);
  for (int k = 0; k < static_cast<int>(settings::adjpop_fission_nuclides.size());
       ++k) {
    const auto& name = settings::adjpop_fission_nuclides[k];
    auto it = data::nuclide_map.find(name);
    if (it == data::nuclide_map.end())
      fatal_error(fmt::format("<adjoint_populations> photoneutron fission "
                              "nuclide {} is not in the model.",
        name));
    nuclide_bin[it->second] = k;
  }
  thread_ew.assign(static_cast<size_t>(n_threads) * n_ebin_bins(), 0.0);
  batch_ew.assign(n_ebin_bins(), 0.0);
  batches_ew.clear();
  if (probes_on()) {
    if (!settings::adjpop_photoneutrons)
      fatal_error("<adjoint_populations> photoneutron_probe_energies require "
                  "photoneutrons to be on.");
    if (!settings::photon_transport)
      fatal_error("<adjoint_populations> photoneutron_probe_energies require "
                  "photon transport.");
    const int photon = ParticleType::photon().transport_index();
    const double e_max =
      std::min(data::energy_max[photon], settings::energy_max[photon]);
    if (settings::adjpop_probe_energies.front() <
          settings::energy_cutoff[photon] ||
        settings::adjpop_probe_energies.back() > e_max) {
      fatal_error(fmt::format("<adjoint_populations> photoneutron probe "
                              "energies must lie between the photon energy "
                              "cutoff and {:.6g} eV.",
        e_max));
    }
  }
  thread_pw.assign(
    static_cast<size_t>(n_threads) * n_probe_depth_bins(), 0.0);
  batch_pw.assign(n_probe_depth_bins(), 0.0);
  batches_pw.clear();
  batch_pf.assign(probes_on() ? n_nbins() : 0, 0.0);
  batches_pf.clear();
  thread_probe_roots.assign(n_threads, {});
  if (rays_requested() && !probes_on())
    fatal_error("<adjoint_populations> ray probes require "
                "photoneutron_probe_energies.");
  ray_lines.clear();
  ray_comb.clear();
  if (rays_requested()) {
    // The probe lines, each interval subdivided linearly in energy
    const auto& e = settings::adjpop_probe_energies;
    const int r = settings::adjpop_ray_refinement;
    for (size_t k = 0; k + 1 < e.size(); ++k)
      for (int i = 0; i < r; ++i)
        ray_lines.push_back(e[k] + (e[k + 1] - e[k]) * i / r);
    ray_lines.push_back(e.back());
    // The comb needs the lines' channels, which init_rays() builds; its
    // first dimension does not depend on the comb
    ray_comb = settings::adjpop_ray_neutron_energies;
    if (ray_comb.empty())
      ray_comb.assign(2, 1.0); // placeholder so that rays_on() holds
    init_rays();
    if (settings::adjpop_ray_neutron_energies.empty())
      build_ray_comb(settings::adjpop_ray_comb_points);
  }
  ray_cast_target.assign(n_nbins(), 0.0);
  ray_root_target.assign(n_nbins(), 0.0);
  ray_site_w.assign(n_nbins(), 1.0);
  probe_photon_target.assign(n_nbins(), 0.0);
  probe_root_target.assign(static_cast<size_t>(n_nbins()) * n_lines(), 0.0);
  probe_site_w.assign(static_cast<size_t>(n_nbins()) * n_lines(), 1.0);
  thread_rw.assign(static_cast<size_t>(n_threads) * n_ray_line_bins(), 0.0);
  thread_rj.assign(static_cast<size_t>(n_threads) * n_ray_comb_bins(), 0.0);
  batch_rw.assign(n_ray_line_bins(), 0.0);
  batch_rj.assign(n_ray_comb_bins(), 0.0);
  batches_rw.clear();
  batches_rj.clear();
  batch_rf.assign(rays_on() ? n_nbins() : 0, 0.0);
  batches_rf.clear();
  thread_tree_buf.assign(n_threads, vector<double>(n_depth(), 0.0));
  thread_tree_active.assign(n_threads, 0);
  thread_branch_raw.assign(static_cast<size_t>(n_threads) * 2, 0.0);
  thread_branch_n.assign(static_cast<size_t>(n_threads) * 2, 0);
  batch_w.assign(n_bins(), 0.0);
  batch_wt.assign(n_bins(), 0.0);
  batches_w.clear();
  batches_wt.clear();
  n_batches_recorded = 0;
  n_histories = 0;
}

void reset_generation()
{
  simulation::adjpop_on = settings::adjpop_n_generation > 0 &&
                          simulation::current_batch > settings::n_inactive;
  if (!simulation::adjpop_on)
    return;
  for (auto& v : thread_roots)
    v.clear();
  for (auto& v : thread_fissions)
    v.clear();
  std::fill(thread_w.begin(), thread_w.end(), 0.0);
  std::fill(thread_wt.begin(), thread_wt.end(), 0.0);
  std::fill(thread_ew.begin(), thread_ew.end(), 0.0);
  std::fill(thread_pw.begin(), thread_pw.end(), 0.0);
  std::fill(thread_rw.begin(), thread_rw.end(), 0.0);
  std::fill(thread_rj.begin(), thread_rj.end(), 0.0);
  for (auto& v : thread_probe_roots)
    v.clear();
  std::fill(thread_branch_raw.begin(), thread_branch_raw.end(), 0.0);
  std::fill(thread_branch_n.begin(), thread_branch_n.end(), 0);
}

void record_fission(Particle& p, int i_nuclide, const Reaction& rx)
{
  if (!simulation::adjpop_on || p.shadow_depth() >= 0)
    return;
  const auto& micro = p.neutron_xs(i_nuclide);
  if (!(micro.total > 0.0) || !(micro.fission > 0.0))
    return;
  FissionRecord f;
  f.r = p.r();
  f.E = p.E();
  f.time = p.time();
  f.wgt = p.wgt() / simulation::keff;
  f.sigma_f_to_t = micro.fission / micro.total;
  f.rx = &rx;
  f.i_nuclide = i_nuclide;
  f.seed_id = combine_ids(
    {generation_key(), p.id(), p.n_tracks(), p.n_event(), i_nuclide, 101});
  thread_fissions[thread_num()].push_back(f);
}

bool record_photoneutron(
  Particle& p, double wgt, Direction u, double E, int delayed_group)
{
  if (!simulation::adjpop_on || !settings::adjpop_photoneutrons ||
      p.shadow_depth() >= 0)
    return false;
  // Only photoneutrons create_secondary() would have banked; the rest are
  // left to it, so that its energy bookkeeping is unchanged
  const int neutron = ParticleType::neutron().transport_index();
  if (E < settings::energy_cutoff[neutron] || E > settings::energy_max[neutron])
    return false;
  Root r;
  r.r = p.r();
  r.u = u;
  r.E = E;
  r.time = p.time();
  // Divided by k like every fission site, so that its importance is on the
  // same footing as the fission roots'
  r.wgt = wgt / simulation::keff;
  r.cls = CLASS_PHOTONEUTRON;
  r.tag = std::clamp(delayed_group, 0, N_TAG - 1);
  r.ebin = energy_group(p, E);
  r.seed_id = combine_ids(
    {generation_key(), p.id(), p.n_tracks(), p.n_event(), bits(E), 202});
  thread_roots[thread_num()].push_back(r);
  return true;
}

bool record_probe_photoneutron(Particle& p, double wgt, Direction u, double E)
{
  if (p.shadow_depth() < 0 || !p.type().is_photon() ||
      tag_class(p.shadow_tag()) != CLASS_PROBE)
    return false;
  // Kept or dropped as record_photoneutron() keeps or drops a driver
  // photoneutron
  const int neutron = ParticleType::neutron().transport_index();
  if (E < settings::energy_cutoff[neutron] || E > settings::energy_max[neutron])
    return true;
  // With rays on, the uncollided photoneutrons are the rays'
  if (rays_on() && p.shadow_tag() % N_TAG == 0)
    return true;
  Root r;
  r.r = p.r();
  r.u = u;
  r.E = E;
  r.time = p.time();
  // The probe's weight already carries 1/k
  r.wgt = wgt;
  r.cls = CLASS_PROBE;
  r.tag = p.shadow_tag() % N_TAG;
  r.ebin = tag_ebin(p.shadow_tag());
  r.seed_id =
    combine_ids({p.id(), p.n_tracks(), p.n_event(), bits(E), bits(wgt), 505});
  thread_probe_roots[thread_num()].push_back(r);
  return true;
}

void mark_scattered(Particle& p)
{
  if (p.shadow_depth() >= 0)
    p.shadow_tag() = secondary_photon_tag(p.shadow_tag());
}

void mark_coherent(Particle& p)
{
  if (p.shadow_depth() >= 0 && tag_class(p.shadow_tag()) == CLASS_PROBE &&
      p.shadow_tag() % N_TAG == 0)
    p.shadow_tag() += 1;
}

int secondary_photon_tag(int shadow_tag)
{
  if (tag_class(shadow_tag) != CLASS_PROBE)
    return shadow_tag;
  return shadow_tag - shadow_tag % N_TAG + 2;
}

void create_tree_sites(Particle& p, int i_nuclide, const Reaction& rx)
{
  const int cls = tag_class(p.shadow_tag());
  const int tag = p.shadow_tag() % N_TAG;
  const int ebin = tag_ebin(p.shadow_tag());
  const double w_site = target_weight(cls, ebin);

  // Expected sites of weight w_site: the same expected banked weight as the
  // driver's unit-weight sites, carried by proportionally fewer sites
  const auto& micro = p.neutron_xs(i_nuclide);
  const double nu_t =
    p.wgt() / simulation::keff * micro.nu_fission / micro.total / w_site;
  int nu = static_cast<int>(nu_t);
  if (prn(p.current_seed()) <= (nu_t - nu))
    ++nu;
  if (nu == 0)
    return;

  const int depth = p.shadow_depth() + 1;
  // IFP's generation-time weight: the lifetime the depth-0 ancestor had
  // reached at the fission that started this line
  const double t0 = (p.shadow_depth() == 0) ? p.lifetime() : p.shadow_t0();
  const int neutron = ParticleType::neutron().transport_index();

  for (int i = 0; i < nu; ++i) {
    SourceSite site;
    site.r = p.r();
    site.particle = ParticleType::neutron();
    site.time = p.time();
    site.wgt = w_site;
    site.surf_id = 0;
    sample_fission_neutron(i_nuclide, rx, &site, p);
    if (site.E > settings::energy_max[neutron])
      continue;
    if (site.delayed_group > 0 && site.time > settings::time_cutoff[neutron])
      continue;

    score_tree(cls, tag, ebin, depth, w_site, t0);
    if (depth < settings::adjpop_n_generation) {
      site.shadow_depth = depth;
      site.shadow_tag = p.shadow_tag();
      site.shadow_t0 = t0;
      site.wgt_born = p.wgt_born();
      site.wgt_ww_born = p.wgt_ww_born();
      site.n_split = p.n_split();
      p.local_secondary_bank().push_back(site);
    }
  }
}

bool tree_makes_photons(const Particle& p)
{
  if (!settings::adjpop_perturbed_importance || p.shadow_depth() < 1)
    return false;
  const int cls = tag_class(p.shadow_tag());
  return cls == CLASS_FISSION || cls == CLASS_DELAYED;
}

bool branch_photoneutron(Particle& p, double& wgt, int& tag)
{
  const int cls = tag_class(p.shadow_tag());
  // Only fission- and delayed-root trees make photons (tree_makes_photons)
  if (cls != CLASS_FISSION && cls != CLASS_DELAYED)
    return false;
  const int b = cls + CLASS_FISSION_BRANCH;
  const size_t slot = static_cast<size_t>(thread_num()) * 2 + cls;
  thread_branch_raw[slot] += wgt;
  const double target = site_w[b];
  if (wgt < target) {
    if (prn(p.current_seed()) >= wgt / target)
      return false;
    wgt = target;
  }
  ++thread_branch_n[slot];
  tag = b * N_TAG + p.shadow_tag() % N_TAG;
  return true;
}

void run_shadow_pass()
{
  if (!simulation::adjpop_on)
    return;

  // Merge the driver's records in a deterministic order
  vector<FissionRecord> fissions;
  for (const auto& v : thread_fissions)
    fissions.insert(fissions.end(), v.begin(), v.end());
  std::sort(fissions.begin(), fissions.end(),
    [](const FissionRecord& a, const FissionRecord& b) {
      return a.seed_id < b.seed_id;
    });
  vector<Root> photo;
  for (const auto& v : thread_roots)
    photo.insert(photo.end(), v.begin(), v.end());
  std::sort(photo.begin(), photo.end(),
    [](const Root& a, const Root& b) { return a.seed_id < b.seed_id; });

  // Each population's raw weight this generation, and from it the target
  // weight that leaves about root_fraction * n_particles roots
  double raw[N_CLASS] {0.0, 0.0, 0.0};
  for (int64_t i = 0; i < simulation::fission_bank.size(); ++i)
    raw[CLASS_FISSION] += simulation::fission_bank[i].wgt;
  {
    vector<std::pair<Root, double>> none;
    for (const auto& f : fissions)
      emit_delayed(f, none, raw[CLASS_DELAYED], false);
  }
  for (const auto& r : photo)
    raw[CLASS_PHOTONEUTRON] += r.wgt;
  const double n_work = static_cast<double>(simulation::work_per_rank);
  const double m_target = std::max(1.0, settings::adjpop_root_fraction * n_work);
  const double m_probe_root =
    std::max(1.0, root_fraction_or_default(settings::adjpop_probe_root_fraction) * n_work);
  const double m_ray_root =
    std::max(1.0, root_fraction_or_default(settings::adjpop_ray_root_fraction) * n_work);
  for (int c = 0; c < N_CLASS; ++c) {
    last_raw_weight[c] = raw[c];
    if (raw[c] > 0.0)
      site_w[c] = raw[c] / m_target;
  }
  // First guess of the branch targets, before any branch has been seen: the
  // driver's photoneutron weight per fission-bank weight, spread over the
  // depths that branch, so that each branched class makes about m_target
  // branches per pass. Afterwards they follow the measured branch weight.
  if (settings::adjpop_perturbed_importance) {
    for (int c = 0; c < 2; ++c) {
      const int b = c + CLASS_FISSION_BRANCH;
      if (site_w[b] == 0.0 && raw[CLASS_PHOTONEUTRON] > 0.0 &&
          raw[CLASS_FISSION] > 0.0) {
        site_w[b] = site_w[c] * raw[CLASS_PHOTONEUTRON] / raw[CLASS_FISSION] *
                    std::max(1, settings::adjpop_n_generation - 1);
      }
      // No photoneutron seen yet: any positive target is unbiased, and the
      // tree's own is safe until the branch weight has been measured
      if (site_w[b] == 0.0)
        site_w[b] = site_w[c];
    }
  }

  // The surviving roots of all three populations
  vector<std::pair<Root, double>> roots;
  for (int64_t i = 0; i < simulation::fission_bank.size(); ++i) {
    const SourceSite& s = simulation::fission_bank[i];
    Root r;
    r.r = s.r;
    r.u = s.u;
    r.E = s.E;
    r.time = s.time;
    r.wgt = s.wgt;
    r.cls = CLASS_FISSION;
    r.tag = std::clamp(s.delayed_group, 0, N_TAG - 1);
    // Keyed on the site's identity, not its index in the bank
    r.seed_id = combine_ids({generation_key(), s.parent_id, s.progeny_id, 303});
    const double w = roulette(r.wgt, CLASS_FISSION, r.seed_id);
    if (w > 0.0)
      roots.emplace_back(r, w);
  }
  {
    double dummy = 0.0;
    for (const auto& f : fissions)
      emit_delayed(f, roots, dummy, true);
  }
  for (const auto& r : photo) {
    const double w = roulette(r.wgt, CLASS_PHOTONEUTRON, r.seed_id);
    if (w > 0.0)
      roots.emplace_back(r, w);
  }
  for (int c = 0; c < N_CLASS; ++c)
    last_roots[c] = 0;
  for (const auto& rw : roots)
    ++last_roots[rw.first.cls];

  // Shadow trees score nothing but their own populations: every tally is
  // suspended for the pass.
  vector<int>* lists[] {&model::active_tallies, &model::active_analog_tallies,
    &model::active_tracklength_tallies,
    &model::active_timed_tracklength_tallies, &model::active_collision_tallies,
    &model::active_meshsurf_tallies, &model::active_surface_tallies,
    &model::active_pulse_height_tallies};
  vector<int> saved[8];
  for (int i = 0; i < 8; ++i)
    saved[i].swap(*lists[i]);

  const auto n = static_cast<int64_t>(roots.size());
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i)
    run_one_tree(roots[i].first, roots[i].second);

  // Probes, after every other tree and in loops of their own, so that the
  // other populations are grown and summed exactly as without them
  if (probes_on()) {
    const int nb = n_nbins();
    vector<double> raw_photon(nb, 0.0);
    vector<int64_t> n_event(nb, 0);
    for (const auto& f : fissions) {
      const double wf = f.wgt * f.sigma_f_to_t;
      const int b = nuclide_bin_of(f.i_nuclide);
      raw_photon[b] += wf * n_lines();
      ++n_event[b];
      batch_pf[b] += wf;
    }
    // Probe photons are cheap next to the trees their photoneutrons grow,
    // which are rouletted to their own budget. Both are shared among the
    // nuclide bins; an event may be split into several photons.
    const double m_photon = std::max(1.0, settings::adjpop_probe_fraction * n_work);
    probe_photon_target = split_targets(raw_photon, n_event, m_photon);
    vector<Probe> probes;
    for (const auto& f : fissions)
      emit_probe(f, probes);
    last_probe_photons = static_cast<int64_t>(probes.size());

    const auto n_probes = static_cast<int64_t>(probes.size());
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_probes; ++i)
      run_one_probe(probes[i]);

    vector<Root> probe_roots;
    for (const auto& v : thread_probe_roots)
      probe_roots.insert(probe_roots.end(), v.begin(), v.end());
    std::sort(probe_roots.begin(), probe_roots.end(),
      [](const Root& a, const Root& b) { return a.seed_id < b.seed_id; });
    // The budget is shared among the nuclide bins, then within a bin among
    // its lines, half in proportion to their photoneutron weight and half
    // equally, so that a line that makes few photoneutrons still gets roots
    // of its own; each root is rouletted to its line's target
    const int K = n_lines();
    vector<double> raw_probe(nb, 0.0), raw_line(static_cast<size_t>(nb) * K, 0.0);
    vector<int64_t> n_probe(nb, 0), n_line(static_cast<size_t>(nb) * K, 0);
    for (const auto& r : probe_roots) {
      raw_probe[r.ebin / K] += r.wgt;
      ++n_probe[r.ebin / K];
      raw_line[r.ebin] += r.wgt;
      ++n_line[r.ebin];
    }
    const vector<double> bin_target =
      share_targets(raw_probe, n_probe, m_probe_root);
    for (int b = 0; b < nb; ++b) {
      const double m_b = (bin_target[b] > 0.0) ? raw_probe[b] / bin_target[b]
                                               : static_cast<double>(n_probe[b]);
      int live = 0;
      for (int k = 0; k < K; ++k)
        live += n_line[static_cast<size_t>(b) * K + k] > 0;
      for (int k = 0; k < K; ++k) {
        const size_t i = static_cast<size_t>(b) * K + k;
        double t = 0.0;
        if (n_line[i] > 0) {
          const double m_k =
            m_b * (0.5 * raw_line[i] / raw_probe[b] + 0.5 / live);
          if (static_cast<double>(n_line[i]) > m_k)
            t = raw_line[i] / m_k;
        }
        probe_root_target[i] = t;
        probe_site_w[i] = (t > 0.0)         ? t
                          : (n_line[i] > 0) ? raw_line[i] / n_line[i]
                                            : 1.0;
      }
    }
    vector<std::pair<Root, double>> kept;
    for (const auto& r : probe_roots) {
      const double w =
        roulette_to(r.wgt, probe_root_target[r.ebin], r.seed_id);
      if (w > 0.0)
        kept.emplace_back(r, w);
    }
    last_probe_roots = static_cast<int64_t>(kept.size());
    const auto n_kept = static_cast<int64_t>(kept.size());
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_kept; ++i)
      run_one_tree(kept[i].first, kept[i].second);
  }

  // Ray probes, after everything else
  if (rays_on()) {
    const int nb = n_nbins();
    vector<double> raw_cast(nb, 0.0);
    vector<int64_t> n_cast(nb, 0);
    for (const auto& f : fissions) {
      const double wf = f.wgt * f.sigma_f_to_t;
      const int b = nuclide_bin_of(f.i_nuclide);
      raw_cast[b] += wf;
      ++n_cast[b];
      batch_rf[b] += wf;
    }
    const double m_ray = std::max(1.0,
      settings::adjpop_ray_fraction * static_cast<double>(simulation::work_per_rank));
    ray_cast_target = split_targets(raw_cast, n_cast, m_ray);
    // Which events cast rays, how many and with what weight (decided in
    // order)
    struct Cast {
      int64_t i;
      int64_t sid;
      double w;
    };
    vector<Cast> casts;
    for (int64_t i = 0; i < static_cast<int64_t>(fissions.size()); ++i) {
      const auto& f = fissions[i];
      const int64_t sid = combine_ids({f.seed_id, 707});
      double w;
      const int n = split_to(f.wgt * f.sigma_f_to_t,
        ray_cast_target[nuclide_bin_of(f.i_nuclide)], sid, w);
      for (int c = 0; c < n; ++c)
        casts.push_back({i, copy_id(sid, c), w});
    }
    last_rays = static_cast<int64_t>(casts.size());
    const auto n_casts = static_cast<int64_t>(casts.size());
    vector<RayRoot> rroots(n_casts);
    vector<char> made(n_casts, 0);
    int64_t n_seg = 0;
#pragma omp parallel for schedule(static) reduction(+ : n_seg)
    for (int64_t i = 0; i < n_casts; ++i) {
      const auto& f = fissions[casts[i].i];
      made[i] = cast_ray(f, casts[i].w, casts[i].sid, rroots[i], n_seg);
    }
    n_ray_segments += n_seg;
    vector<double> raw_r(nb, 0.0);
    vector<int64_t> n_r(nb, 0);
    for (int64_t i = 0; i < n_casts; ++i)
      if (made[i]) {
        raw_r[rroots[i].nbin] += rroots[i].root.wgt;
        ++n_r[rroots[i].nbin];
      }
    ray_root_target = share_targets(raw_r, n_r, m_ray_root);
    // A bin's trees bank at its target, or at its mean root weight if it
    // keeps all its roots
    for (int b = 0; b < nb; ++b)
      ray_site_w[b] = (ray_root_target[b] > 0.0) ? ray_root_target[b]
                      : (n_r[b] > 0)            ? raw_r[b] / n_r[b]
                                                : 1.0;
    vector<std::pair<int64_t, double>> grow;
    for (int64_t i = 0; i < n_casts; ++i) {
      if (!made[i])
        continue;
      const double w = roulette_to(rroots[i].root.wgt,
        ray_root_target[rroots[i].nbin], rroots[i].root.seed_id);
      if (w > 0.0)
        grow.emplace_back(i, w);
    }
    last_ray_roots = static_cast<int64_t>(grow.size());
    const auto n_grow = static_cast<int64_t>(grow.size());
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n_grow; ++i) {
      // The factors are per unit raw weight; the tree carries its rouletted
      // weight, whose expected value is the raw weight
      run_one_ray_tree(rroots[grow[i].first], grow[i].second);
    }
  }

  for (int i = 0; i < 8; ++i)
    saved[i].swap(*lists[i]);

  // Branch weight of this pass sets the branch targets of the next
  if (settings::adjpop_perturbed_importance) {
    for (int c = 0; c < 2; ++c) {
      const int b = c + CLASS_FISSION_BRANCH;
      double raw_b = 0.0;
      int64_t n_b = 0;
      for (int t = 0; t < num_threads(); ++t) {
        raw_b += thread_branch_raw[static_cast<size_t>(t) * 2 + c];
        n_b += thread_branch_n[static_cast<size_t>(t) * 2 + c];
      }
      last_raw_weight[b] = raw_b;
      last_roots[b] = n_b;
      if (raw_b > 0.0)
        site_w[b] = raw_b / m_target;
    }
  }

  // Reduce the slabs, in thread order, into this batch
  const int nb = n_bins();
  for (int t = 0; t < num_threads(); ++t) {
    for (int i = 0; i < nb; ++i) {
      batch_w[i] += thread_w[static_cast<size_t>(t) * nb + i];
      batch_wt[i] += thread_wt[static_cast<size_t>(t) * nb + i];
    }
  }
  const int ne = n_ebin_bins();
  for (int t = 0; t < num_threads(); ++t)
    for (int i = 0; i < ne; ++i)
      batch_ew[i] += thread_ew[static_cast<size_t>(t) * ne + i];
  const int np = n_probe_depth_bins();
  for (int t = 0; t < num_threads(); ++t)
    for (int i = 0; i < np; ++i)
      batch_pw[i] += thread_pw[static_cast<size_t>(t) * np + i];
  const int nrl = n_ray_line_bins(), nrc = n_ray_comb_bins();
  for (int t = 0; t < num_threads(); ++t) {
    for (int i = 0; i < nrl; ++i)
      batch_rw[i] += thread_rw[static_cast<size_t>(t) * nrl + i];
    for (int i = 0; i < nrc; ++i)
      batch_rj[i] += thread_rj[static_cast<size_t>(t) * nrc + i];
  }
}

void finalize_batch()
{
  if (settings::adjpop_n_generation <= 0 ||
      simulation::current_batch <= settings::n_inactive)
    return;
#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    vector<double> rw(batch_w.size()), rwt(batch_wt.size());
    mpi::reduce<double>(
      batch_w.data(), rw.data(), batch_w.size(), MPI_SUM, 0, mpi::intracomm);
    mpi::reduce<double>(
      batch_wt.data(), rwt.data(), batch_wt.size(), MPI_SUM, 0, mpi::intracomm);
    if (mpi::master) {
      batch_w.swap(rw);
      batch_wt.swap(rwt);
    }
    if (!batch_ew.empty()) {
      vector<double> rew(batch_ew.size());
      mpi::reduce<double>(batch_ew.data(), rew.data(), batch_ew.size(), MPI_SUM,
        0, mpi::intracomm);
      if (mpi::master)
        batch_ew.swap(rew);
    }
    if (!batch_pw.empty()) {
      vector<double> rpw(batch_pw.size()), rpf(batch_pf.size());
      mpi::reduce<double>(batch_pw.data(), rpw.data(), batch_pw.size(), MPI_SUM,
        0, mpi::intracomm);
      mpi::reduce<double>(batch_pf.data(), rpf.data(), batch_pf.size(), MPI_SUM,
        0, mpi::intracomm);
      if (mpi::master) {
        batch_pw.swap(rpw);
        batch_pf.swap(rpf);
      }
    }
  }
#endif
  if (mpi::master) {
    batches_w.insert(batches_w.end(), batch_w.begin(), batch_w.end());
    batches_wt.insert(batches_wt.end(), batch_wt.begin(), batch_wt.end());
    batches_ew.insert(batches_ew.end(), batch_ew.begin(), batch_ew.end());
    batches_pw.insert(batches_pw.end(), batch_pw.begin(), batch_pw.end());
    batches_pf.insert(batches_pf.end(), batch_pf.begin(), batch_pf.end());
    batches_rw.insert(batches_rw.end(), batch_rw.begin(), batch_rw.end());
    batches_rj.insert(batches_rj.end(), batch_rj.begin(), batch_rj.end());
    batches_rf.insert(batches_rf.end(), batch_rf.begin(), batch_rf.end());
    ++n_batches_recorded;
  }
  std::fill(batch_w.begin(), batch_w.end(), 0.0);
  std::fill(batch_wt.begin(), batch_wt.end(), 0.0);
  std::fill(batch_ew.begin(), batch_ew.end(), 0.0);
  std::fill(batch_pw.begin(), batch_pw.end(), 0.0);
  std::fill(batch_pf.begin(), batch_pf.end(), 0.0);
  std::fill(batch_rw.begin(), batch_rw.end(), 0.0);
  std::fill(batch_rj.begin(), batch_rj.end(), 0.0);
  std::fill(batch_rf.begin(), batch_rf.end(), 0.0);
}

double probe_trigger_ratio(int& worst_bin, double& worst_energy)
{
  worst_bin = -1;
  worst_energy = 0.0;
  int worst_line = -1;
  if (!(settings::adjpop_probe_trigger > 0.0) || !probes_on())
    return 0.0;
  const int nb = n_batches_recorded;
  if (nb < 2)
    return INFTY;
  const int L = settings::adjpop_n_generation;
  const size_t nd = n_depth(), K = n_lines(), nn = n_nbins();
  // The trigger is checked at every ray line (the probe lines if no rays),
  // the probes' collided part interpolated per batch from the probe lines
  const int r = rays_on() ? settings::adjpop_ray_refinement : 1;
  const size_t KR = rays_on() ? static_cast<size_t>(n_ray_lines()) : K;

  // I_F per batch at depth L
  vector<double> y(nb, 0.0);
  for (int b = 0; b < nb; ++b)
    for (int t = 0; t < N_TAG; ++t)
      y[b] +=
        batches_w[static_cast<size_t>(b) * n_bins() + bin(CLASS_FISSION, t, L)];
  double my = 0.0;
  for (double v : y)
    my += v;
  my /= nb;
  if (!(my > 0.0))
    return INFTY;

  // The listed nuclides, or the single bin if none is listed
  const int n_check =
    settings::adjpop_fission_nuclides.empty()
      ? 1
      : static_cast<int>(settings::adjpop_fission_nuclides.size());
  double worst = 0.0;
  vector<double> x(nb);
  vector<double> rr(KR), s(KR);
  auto probe_sum = [&](size_t jb, size_t k) {
    double v = 0.0;
    for (int l = 0; l < N_PROBE_LABEL; ++l)
      v += batches_pw[((jb * K + k) * N_PROBE_LABEL + l) * nd + L];
    return v;
  };
  for (int j = 0; j < n_check; ++j) {
    double peak = 0.0;
    for (size_t i = 0; i < KR; ++i) {
      // Direct + scattered: the rays (if on) and every probe label
      const size_t k = i / r;
      const double t = static_cast<double>(i % r) / r;
      double mx = 0.0;
      for (int b = 0; b < nb; ++b) {
        const size_t jb = static_cast<size_t>(b) * nn + j;
        double v = probe_sum(jb, k) * (1.0 - t);
        if (t > 0.0)
          v += probe_sum(jb, k + 1) * t;
        if (rays_on())
          v += batches_rw[(jb * KR + i) * nd + L];
        x[b] = v;
        mx += v;
      }
      mx /= nb;
      rr[i] = mx / my;
      // Delta method for the ratio of means
      double ss = 0.0;
      for (int b = 0; b < nb; ++b) {
        const double z = (x[b] - rr[i] * y[b]) / my;
        ss += z * z;
      }
      s[i] = std::sqrt(ss / (nb - 1) / nb);
      peak = std::max(peak, rr[i]);
    }
    if (!(peak > 0.0)) {
      worst_bin = j;
      return INFTY;
    }
    for (size_t i = 0; i < KR; ++i) {
      const double ref =
        std::max(rr[i], settings::adjpop_probe_trigger_floor * peak);
      const double ratio = s[i] / (settings::adjpop_probe_trigger * ref);
      if (ratio > worst) {
        worst = ratio;
        worst_bin = j;
        worst_line = static_cast<int>(i);
      }
    }
  }
  if (worst_line >= 0)
    worst_energy = rays_on() ? ray_lines[worst_line]
                             : settings::adjpop_probe_energies[worst_line];
  return worst;
}

void write_results(hid_t file_id)
{
  if (settings::adjpop_n_generation <= 0 || n_batches_recorded == 0)
    return;
  hid_t group = create_group(file_id, "adjoint_populations");
  write_dataset(group, "n_generation", settings::adjpop_n_generation);
  write_dataset(group, "n_batches", n_batches_recorded);
  write_dataset(group, "n_class", N_SCORE_CLASS);
  write_dataset(group, "n_tag", N_TAG);
  write_dataset(
    group, "photoneutrons", static_cast<int>(settings::adjpop_photoneutrons));
  write_dataset(group, "perturbed_importance",
    static_cast<int>(settings::adjpop_perturbed_importance));
  // [batch][class][tag][depth], flattened
  write_dataset(group, "weight", batches_w);
  write_dataset(group, "weight_t0", batches_wt);
  vector<double> sw(site_w, site_w + N_SCORE_CLASS);
  write_dataset(group, "site_weight", sw);
  vector<int64_t> nr(last_roots, last_roots + N_SCORE_CLASS);
  write_dataset(group, "n_roots_last_generation", nr);
  vector<double> rw(last_raw_weight, last_raw_weight + N_SCORE_CLASS);
  write_dataset(group, "raw_weight_last_generation", rw);
  write_dataset(group, "n_histories", n_histories);
  write_dataset(group, "root_fraction", settings::adjpop_root_fraction);
  if (tagging_on()) {
    const size_t nd = n_depth(), ne = n_egroups(), nn = n_nbins();
    const size_t per_batch = nn * ne * nd;
    if (n_ebins() > 0) {
      // [batch][energy group][depth], flattened: summed over nuclide bins
      vector<double> ew(static_cast<size_t>(n_batches_recorded) * ne * nd, 0.0);
      for (size_t b = 0; b < static_cast<size_t>(n_batches_recorded); ++b)
        for (size_t n = 0; n < nn; ++n)
          for (size_t i = 0; i < ne * nd; ++i)
            ew[b * ne * nd + i] += batches_ew[b * per_batch + n * ne * nd + i];
      write_dataset(
        group, "photoneutron_energy_bins", settings::adjpop_energy_bins);
      write_dataset(group, "photoneutron_energy_variable",
        std::string(settings::adjpop_energy_variable == 0 ? "photon_birth"
                                                          : "photoneutron"));
      write_dataset(group, "photoneutron_energy_weight", ew);
    }
    if (!settings::adjpop_fission_nuclides.empty()) {
      // [batch][nuclide bin][energy group][depth], flattened; the last
      // nuclide bin holds every other nuclide (and non-fission photons)
      std::string names;
      for (const auto& n : settings::adjpop_fission_nuclides)
        names += n + " ";
      names += "other";
      write_dataset(group, "photoneutron_fission_nuclides", names);
      write_dataset(group, "photoneutron_nuclide_weight", batches_ew);
    }
  }
  if (probes_on()) {
    // Nuclide bins as for the photoneutron roots: the listed nuclides and
    // "other", or "all" if none is listed
    std::string names;
    for (const auto& n : settings::adjpop_fission_nuclides)
      names += n + " ";
    names += settings::adjpop_fission_nuclides.empty() ? "all" : "other";
    write_dataset(group, "probe_fission_nuclides", names);
    write_dataset(group, "probe_energies", settings::adjpop_probe_energies);
    // [batch][nuclide bin][line][label][depth], flattened; label 0
    // uncollided (empty with rays on: see ray_weight), 1 after coherent
    // scattering only, 2 after an energy-changing collision
    write_dataset(group, "probe_n_labels", N_PROBE_LABEL);
    write_dataset(group, "probe_weight", batches_pw);
    // [batch][nuclide bin]: the probed fission weight, sum of w/k
    // sigma_f/sigma_t over the recorded fission events
    write_dataset(group, "probe_fission_weight", batches_pf);
    // The probe photons' targets [nuclide bin], then the probe roots'
    // [nuclide bin][line]
    vector<double> pt(probe_photon_target);
    pt.insert(pt.end(), probe_root_target.begin(), probe_root_target.end());
    write_dataset(group, "probe_site_weight", pt);
    write_dataset(group, "n_probe_photons_last_generation", last_probe_photons);
    write_dataset(group, "n_probe_roots_last_generation", last_probe_roots);
    write_dataset(group, "n_probe_histories", n_probe_histories);
    write_dataset(group, "probe_root_fraction",
      root_fraction_or_default(settings::adjpop_probe_root_fraction));
    if (settings::adjpop_probe_trigger > 0.0)
      write_dataset(group, "probe_trigger",
        vector<double> {settings::adjpop_probe_trigger,
          settings::adjpop_probe_trigger_floor});
  }
  if (rays_on()) {
    write_dataset(group, "ray_neutron_energies", ray_comb);
    write_dataset(group, "ray_photon_energies", ray_lines);
    write_dataset(group, "ray_refinement", settings::adjpop_ray_refinement);
    // [batch][nuclide bin][line][depth]: uncollided photoneutron importance
    write_dataset(group, "ray_weight", batches_rw);
    // [batch][nuclide bin][comb energy][depth]: the trees by comb energy
    write_dataset(group, "ray_comb_weight", batches_rj);
    // [batch][nuclide bin]: sum of w/k sigma_f/sigma_t over the events
    write_dataset(group, "ray_fission_weight", batches_rf);
    // [2][nuclide bin]: the rays' and the ray roots' roulette targets
    vector<double> tw(ray_cast_target);
    tw.insert(tw.end(), ray_root_target.begin(), ray_root_target.end());
    write_dataset(group, "ray_site_weight", tw);
    write_dataset(group, "ray_allocation",
      std::string(settings::adjpop_ray_allocation == 0 ? "equal" : "fission"));
    write_dataset(group, "n_rays_last_generation", last_rays);
    write_dataset(group, "n_ray_roots_last_generation", last_ray_roots);
    write_dataset(group, "n_ray_segments", n_ray_segments);
    write_dataset(group, "ray_root_fraction",
      root_fraction_or_default(settings::adjpop_ray_root_fraction));
  }
  close_group(group);
}

void clear()
{
  thread_roots.clear();
  thread_fissions.clear();
  thread_w.clear();
  thread_wt.clear();
  thread_branch_raw.clear();
  thread_branch_n.clear();
  batch_w.clear();
  batch_wt.clear();
  batches_w.clear();
  batches_wt.clear();
  thread_ew.clear();
  batch_ew.clear();
  batches_ew.clear();
  thread_pw.clear();
  batch_pw.clear();
  batches_pw.clear();
  batch_pf.clear();
  batches_pf.clear();
  thread_probe_roots.clear();
  probe_photon_target.clear();
  probe_root_target.clear();
  probe_site_w.clear();
  last_probe_photons = 0;
  last_probe_roots = 0;
  n_probe_histories = 0;
  ray_lines.clear();
  ray_comb.clear();
  ray_sigt.clear();
  ray_sign.clear();
  ray_sigt_min.clear();
  ray_channels.clear();
  thread_rw.clear();
  batch_rw.clear();
  batches_rw.clear();
  thread_rj.clear();
  batch_rj.clear();
  batches_rj.clear();
  batch_rf.clear();
  batches_rf.clear();
  thread_tree_buf.clear();
  thread_tree_active.clear();
  ray_cast_target.clear();
  ray_root_target.clear();
  ray_site_w.clear();
  last_rays = last_ray_roots = n_ray_segments = 0;
  nuclide_bin.clear();
  n_batches_recorded = 0;
  n_histories = 0;
  simulation::adjpop_on = false;
  for (int c = 0; c < N_SCORE_CLASS; ++c) {
    site_w[c] = (c < N_CLASS) ? 1.0 : 0.0;
    last_roots[c] = 0;
    last_raw_weight[c] = 0.0;
  }
}

} // namespace adjpop

} // namespace openmc
