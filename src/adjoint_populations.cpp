#include "openmc/adjoint_populations.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fmt/core.h>

#include "openmc/bank.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/openmp_interface.h"
#include "openmc/particle.h"
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
  int nbin = 0;
  if (!settings::adjpop_fission_nuclides.empty()) {
    const int i = photon.fission_nuclide();
    nbin = (i >= 0 && i < static_cast<int>(nuclide_bin.size()))
             ? nuclide_bin[i]
             : n_nbins() - 1;
  }
  return nbin * n_egroups() + group;
}

//! Russian roulette of a root to its population's target weight, drawn on
//! its own key so that no tree replays the number deciding its existence.
//! \return the weight the root is grown with, or 0 if it is killed
double roulette(double w_raw, int cls, int64_t seed_id)
{
  const double target = site_w[cls];
  if (!(w_raw > 0.0))
    return 0.0;
  if (w_raw >= target)
    return w_raw;
  uint64_t s = init_seed(combine_ids({seed_id, 1}), STREAM_TRACKING);
  return (prn(&s) < w_raw / target) ? target : 0.0;
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
  site.shadow_tag =
    root.cls * N_TAG + root.tag + (root.ebin + 1) * N_TAG * N_SCORE_CLASS;
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

  score(root.cls, root.tag, 0, w, 0.0);
  score_ebin(root.ebin, 0, w);
  transport_history_based_single_particle(p);
#pragma omp atomic
  n_histories += p.n_tracks();
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

void create_tree_sites(Particle& p, int i_nuclide, const Reaction& rx)
{
  const int cls = tag_class(p.shadow_tag());
  const int tag = p.shadow_tag() % N_TAG;
  const int ebin = tag_ebin(p.shadow_tag());
  const double w_site = site_w[cls];

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

    score(cls, tag, depth, w_site, t0);
    score_ebin(ebin, depth, w_site);
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
  // weight that leaves about ROOT_FRACTION * n_particles roots
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
  const double m_target = std::max(
    1.0, ROOT_FRACTION * static_cast<double>(simulation::work_per_rank));
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
  }
#endif
  if (mpi::master) {
    batches_w.insert(batches_w.end(), batch_w.begin(), batch_w.end());
    batches_wt.insert(batches_wt.end(), batch_wt.begin(), batch_wt.end());
    batches_ew.insert(batches_ew.end(), batch_ew.begin(), batch_ew.end());
    ++n_batches_recorded;
  }
  std::fill(batch_w.begin(), batch_w.end(), 0.0);
  std::fill(batch_wt.begin(), batch_wt.end(), 0.0);
  std::fill(batch_ew.begin(), batch_ew.end(), 0.0);
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
