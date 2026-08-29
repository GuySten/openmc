//! \file bep.cpp
//! Branched Exact Perturbation. See bep.h for the estimator and lineage.

#include "openmc/bep.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <fmt/core.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"

namespace openmc {

namespace settings {
bool bep_on {false};
int bep_n_generation {10};
} // namespace settings

// Declared in simulation.h; repeated to avoid a circular include.
void transport_history_based_single_particle(Particle& p);

namespace bep {

vector<Perturbation> perturbations;
vector<Tree> trees;
vector<int> cell_ref_tree;
vector<vector<int>> cell_perts;
vector<BranchSite> branch_sites;
vector<double> tau;
vector<double> sum_l;
vector<double> sum_tau;
vector<double> sum_rho;
vector<double> sum_cross;
vector<int64_t> n_pair;
int64_t n_generations {0};
int64_t n_branch_total {0};
double w_branch_total {0.0};

namespace {

inline bool active_generation()
{
  return simulation::current_batch > settings::n_inactive;
}

//! Lower end of the finite difference used for rho_gen.
inline int d_half()
{
  return settings::bep_n_generation / 2;
}

} // namespace

//==============================================================================
// Setup
//==============================================================================

void init()
{
  if (!settings::bep_on)
    return;

  if (settings::run_mode != RunMode::EIGENVALUE)
    fatal_error("<local_perturbation> requires an eigenvalue calculation.");
  if (settings::event_based)
    fatal_error("<local_perturbation> requires history-based transport.");
  if (settings::bep_n_generation < 2) {
    fatal_error("'local_perturbation_n_generation' must be at least 2: the "
                "estimator is a finite difference in depth.");
  }
  if (perturbations.empty())
    fatal_error("<local_perturbation> given with no perturbations defined.");

  std::unordered_set<int32_t> seen_ids;
  for (const auto& p : perturbations) {
    if (!seen_ids.insert(p.id).second)
      fatal_error(fmt::format("Duplicate <local_perturbation> id {}.", p.id));
    if (p.subs.empty()) {
      fatal_error(fmt::format(
        "<local_perturbation> {} has no substitutions.", p.id));
    }
  }

  cell_ref_tree.assign(model::cells.size(), -1);
  cell_perts.assign(model::cells.size(), {});
  trees.clear();

  // Pass 1: resolve every substitution and create one reference tree per
  // distinct touched cell.
  for (size_t ip = 0; ip < perturbations.size(); ++ip) {
    Perturbation& p = perturbations[ip];
    std::unordered_set<int32_t> seen_cells;

    for (auto& s : p.subs) {
      auto c = model::cell_map.find(s.cell_id);
      if (c == model::cell_map.end()) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} not found.", p.id, s.cell_id));
      }
      s.cell_index = c->second;

      if (!seen_cells.insert(s.cell_index).second) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} appears twice; a perturbation "
          "may substitute each cell at most once.",
          p.id, s.cell_id));
      }

      const Cell& cell {*model::cells[s.cell_index]};
      if (cell.type_ != Fill::MATERIAL) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} must be filled with a material; "
          "the swap replaces a material, not a universe or lattice.",
          p.id, s.cell_id));
      }
      if (cell.material_.size() != 1) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} must have exactly one material "
          "(no distribcell material list); the swap is per-cell.",
          p.id, s.cell_id));
      }

      auto m = model::material_map.find(s.mat_id);
      if (m == model::material_map.end()) {
        fatal_error(fmt::format("<local_perturbation> {}: material {} not "
                                "found.",
          p.id, s.mat_id));
      }
      s.mat_index = m->second;

      if (cell_ref_tree[s.cell_index] < 0) {
        cell_ref_tree[s.cell_index] = static_cast<int>(trees.size());
        trees.push_back({-1});
      }
      cell_perts[s.cell_index].push_back(static_cast<int>(ip));
      p.cells.push_back(s.cell_index);
    }

    // A perturbation that changes nothing is the null test: legal, and the
    // first thing to run, so warn rather than abort.
    bool null_test = true;
    for (const auto& s : p.subs) {
      if (model::cells[s.cell_index]->material_[0] != s.mat_index) {
        null_test = false;
        break;
      }
    }
    if (null_test) {
      warning(fmt::format("<local_perturbation> {} substitutes materials that "
                          "are already in place. This is the null test; rho "
                          "must come out exactly zero.",
        p.id));
    }
  }

  // Pass 2: one tree per perturbation, after all reference trees exist.
  for (size_t ip = 0; ip < perturbations.size(); ++ip) {
    perturbations[ip].tree = static_cast<int>(trees.size());
    trees.push_back({static_cast<int>(ip)});
  }

  // Shadow roots start at super_gen == 1 so that all existing
  // `super_gen() <= 0` gates exclude them. Relative depth d = super_gen - 1,
  // so the revival loop needs one extra generation to reach d = L.
  settings::super_n_generation = settings::bep_n_generation + 1;

  int nd = settings::bep_n_generation + 1;
  size_t np = perturbations.size();
  tau.assign(trees.size() * nd, 0.0);
  sum_tau.assign(trees.size() * nd, 0.0);
  sum_l.assign(np * nd, 0.0);
  sum_rho.assign(np, 0.0);
  sum_cross.assign(np * np, 0.0);
  n_pair.assign(np * np, 0);
  branch_sites.clear();

  size_t n_cells_touched = trees.size() - np;
  write_message(
    fmt::format("BEP: {} perturbation(s) over {} cell(s), {} shadow trees, "
                "L = {}.",
      np, n_cells_touched, trees.size(), settings::bep_n_generation),
    5);
}

void reset_generation()
{
  if (!settings::bep_on)
    return;
  std::fill(tau.begin(), tau.end(), 0.0);
  branch_sites.clear();
}

//==============================================================================
// Branch detection (driver side)
//==============================================================================

void maybe_branch(Particle& p, int32_t cell_index)
{
  // Caller has established bep_on, that p is a trunk particle, and that
  // cell_index is touched by at least one perturbation.
  if (!p.type().is_neutron())
    return;
  if (p.wgt() == 0.0)
    return;
  if (!active_generation())
    return; // shadow trees are pure overhead before source convergence

  BranchSite site;
  site.r = p.r();
  site.u = p.u();
  site.E = p.E();
  site.wgt = p.wgt();
  site.time = p.time();
  site.cell = cell_index;

#pragma omp critical(BepBranch)
  {
    branch_sites.push_back(site);
    ++n_branch_total;
    w_branch_total += site.wgt;
  }

  // The driver is deliberately NOT stopped and NOT tagged: it carries on in
  // the reference state exactly as in a stock run, so k, the fission source
  // and all ordinary tallies are untouched. The reference shadow is a
  // separate copy on purpose -- the driver's bank is renormalised and combed
  // every generation while a shadow tree is not.
  //
  // A trunk history that leaves and re-enters branches again, and a trunk
  // history that visits two different perturbed cells branches at each. Both
  // are correct: the estimator is the SLOPE of l(d), and each branch is an
  // independent sample of the same slope, so extra branches add (correlated)
  // statistics without bias. Only an absolute-normalisation estimator would
  // need first-entry-only bookkeeping.
}

void score_site(int tree, int super_gen, double wgt)
{
  int depth = super_gen - 1; // shadow roots live at super_gen == 1
  if (depth < 0 || depth > settings::bep_n_generation)
    return;
#pragma omp atomic
  tau[tau_index(tree, depth)] += wgt;
}

//==============================================================================
// Shadow pass
//==============================================================================

namespace {

void run_one_tree(const BranchSite& site, int tree, uint64_t seed)
{
  Particle p;

  SourceSite root;
  root.r = site.r;
  root.u = site.u;
  root.E = site.E;
  root.wgt = site.wgt;
  root.time = site.time;
  root.particle = ParticleType::neutron();
  root.super_gen = 1; // see the depth-encoding note in bep.h
  root.adjoint_id = -1;
  root.bep_tree = tree;

  p.from_source(&root);
  p.n_progeny() = 0;
  p.n_event() = 0;
  p.n_tracks() = 1;
  p.n_split() = 0;
  p.ww_factor() = 0.0;
  p.wgt_born() = p.wgt();
  p.seeds(0) = seed; // common random numbers across the whole tree family
  p.stream() = STREAM_TRACKING;

  score_site(tree, 1, site.wgt); // the root is this tree's depth-0 weight

  transport_history_based_single_particle(p);

  p.local_secondary_bank().clear();
}

} // namespace

void run_shadow_pass()
{
  if (!settings::bep_on || branch_sites.empty())
    return;

  auto n = static_cast<int64_t>(branch_sites.size());

#pragma omp parallel for schedule(dynamic, 8)
  for (int64_t i = 0; i < n; ++i) {
    const BranchSite& site = branch_sites[i];

    // One seed per branch site, shared by the reference tree and every
    // perturbed tree spawned here, so all of them draw the same numbers
    // wherever they are doing the same physics. That is what makes the null
    // test exactly zero and what correlates perturbations with each other.
    uint64_t seed =
      init_seed(simulation::total_gen * 1000000007LL + i, STREAM_TRACKING);

    run_one_tree(site, cell_ref_tree[site.cell], seed);

    // Only the perturbations that touch THIS cell spawn a tree here. A
    // perturbation whose other cells the history later reaches is picked up
    // by a separate branch there, since the driver stays untagged.
    for (int ip : cell_perts[site.cell]) {
      run_one_tree(site, perturbations[ip].tree, seed);
    }
  }
}

void accumulate_generation()
{
  if (!settings::bep_on || !active_generation())
    return;

#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    // Sum tau across ranks before forming any ratio: the estimator is a ratio
    // of sums, not the mean of per-rank ratios.
    vector<double> reduced(tau.size(), 0.0);
    MPI_Allreduce(tau.data(), reduced.data(), static_cast<int>(tau.size()),
      MPI_DOUBLE, MPI_SUM, mpi::intracomm);
    tau = reduced;
  }
#endif

  int L = settings::bep_n_generation;
  int nd = L + 1;
  int dh = d_half();
  auto np = static_cast<int>(perturbations.size());

  for (size_t i = 0; i < tau.size(); ++i)
    sum_tau[i] += tau[i];

  vector<double> rho(np, 0.0);
  vector<char> usable(np, 0);

  for (int ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];

    double l_L = 0.0, l_h = 0.0;
    bool ok = true;
    for (int d = 0; d <= L; ++d) {
      // Matched reference: the sum of the reference trees over exactly the
      // cells this perturbation touches, i.e. over exactly the branch sites
      // where it spawned a tree.
      double R = 0.0;
      for (int32_t ci : p.cells)
        R += tau[tau_index(cell_ref_tree[ci], d)];
      if (R <= 0.0) {
        ok = false; // no branch site reached this depth this generation
        break;
      }
      // log1p of the correlated difference: forming the difference first
      // preserves the cancellation, and the log makes the importance-ratio
      // constant drop out of the slope exactly.
      double l = std::log1p((tau[tau_index(p.tree, d)] - R) / R);
      sum_l[ip * nd + d] += l;
      if (d == L)
        l_L = l;
      if (d == dh)
        l_h = l;
    }
    if (!ok)
      continue;

    rho[ip] = (l_L - l_h) / static_cast<double>(L - dh);
    usable[ip] = 1;
    sum_rho[ip] += rho[ip];
  }

  // Full cross-product matrix so Python can give an exact uncertainty to any
  // linear combination -- a difference between two sample positions, a
  // finite-difference derivative, a fitted traverse. Counts are per pair
  // because a starved perturbation must not invalidate the others.
  for (int i = 0; i < np; ++i) {
    if (!usable[i])
      continue;
    for (int j = 0; j < np; ++j) {
      if (!usable[j])
        continue;
      sum_cross[i * np + j] += rho[i] * rho[j];
      ++n_pair[i * np + j];
    }
  }
  ++n_generations;
}

//==============================================================================
// Output
//==============================================================================

void write_results(hid_t file_id)
{
  if (!settings::bep_on || n_generations == 0)
    return;

  int nd = settings::bep_n_generation + 1;
  auto np = static_cast<int>(perturbations.size());
  hid_t group = create_group(file_id, "local_perturbation");

  write_dataset(group, "n_generation", settings::bep_n_generation);
  write_dataset(group, "d_half", d_half());
  write_dataset(group, "n_generations", n_generations);
  write_dataset(group, "n_branch", n_branch_total);
  write_dataset(group, "w_branch", w_branch_total);
  write_dataset(group, "n_perturbations", np);

  // Raw sums, not means: normalisation, the linearity check and the fit
  // belong in Python so the analysis can change without a rebuild.
  write_dataset(group, "sum_rho", sum_rho);
  write_dataset(group, "sum_cross", sum_cross);
  vector<int64_t> npair(n_pair.begin(), n_pair.end());
  write_dataset(group, "n_pair", npair);

  vector<int32_t> ids;
  for (const auto& p : perturbations)
    ids.push_back(p.id);
  write_dataset(group, "ids", ids);

  for (int ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];
    hid_t pg = create_group(group, fmt::format("perturbation {}", p.id));
    write_dataset(pg, "index", ip);

    vector<int32_t> cids, mids;
    for (const auto& s : p.subs) {
      cids.push_back(s.cell_id);
      mids.push_back(s.mat_id);
    }
    write_dataset(pg, "cells", cids);
    write_dataset(pg, "materials", mids);

    vector<double> l(sum_l.begin() + ip * nd, sum_l.begin() + (ip + 1) * nd);
    write_dataset(pg, "sum_l", l);

    vector<double> tp(sum_tau.begin() + p.tree * nd,
      sum_tau.begin() + (p.tree + 1) * nd);
    vector<double> tr(nd, 0.0);
    for (int32_t ci : p.cells) {
      int rt = cell_ref_tree[ci];
      for (int d = 0; d < nd; ++d)
        tr[d] += sum_tau[rt * nd + d];
    }
    write_dataset(pg, "sum_tau", tp);
    write_dataset(pg, "sum_tau_ref", tr);

    close_group(pg);
  }

  close_group(group);
}

} // namespace bep
} // namespace openmc
