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
#include "openmc/file_utils.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/openmp_interface.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/tallies/tally.h"
#include "openmc/xml_interface.h"
#include "openmc/simulation.h"

namespace openmc {

namespace settings {
bool bep_on {false};
} // namespace settings

// Declared in simulation.h; repeated to avoid a circular include.
void transport_history_based_single_particle(Particle& p);

namespace bep {

vector<Perturbation> perturbations;
vector<Tree> trees;
vector<int> cell_ref_tree;
vector<vector<int>> cell_perts;
bool branching {false};
vector<vector<BranchSite>> thread_branch_sites;
vector<BranchSite> branch_sites;
vector<double> thread_tau;
vector<double> tau;
vector<double> tau_history;
int64_t n_generations {0};
int64_t n_branch_total {0};
double w_branch_total {0.0};

// Internal linkage. `inline` would be redundant inside an unnamed namespace,
// so it is omitted; forward declarations let run_one_tree() live here with
// the rest of the file-local helpers rather than in a second block further
// down.
namespace {

//! 64-bit mix (the splitmix64 finalizer), so that small, correlated inputs
//! give well-separated seeds.
uint64_t mix(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

//! Seed for the shadow trees spawned at one branch point.
//!
//! Keyed on the branch itself -- the generation and the driver particle's
//! identity -- so it does not depend on the order branch sites happen to be
//! recorded in, which is thread-timing dependent. Getting this wrong does
//! not corrupt anything, it just makes every run give different worths from
//! the same input.
int64_t branch_seed(int64_t particle_id, int64_t n_tracks, int n_event)
{
  uint64_t h = mix(static_cast<uint64_t>(simulation::total_gen));
  h = mix(h ^ static_cast<uint64_t>(particle_id));
  h = mix(h ^ (static_cast<uint64_t>(n_tracks) << 32 |
                static_cast<uint64_t>(n_event)));
  return static_cast<int64_t>(h >> 1);
}

//! Transport one shadow tree rooted at `site` in tree `tree`.
//!
//! Depth bookkeeping rides on the existing super-history fields:
//! create_fission_sites() stamps super_gen on each site it makes and calls
//! score_site() with it, so nothing has to be measured after the fact.
//!
//! `seed_id` is shared by every tree spawned at one branch site, which is
//! what makes the trees common-random-number correlated. ALL streams must be
//! initialised from it, not just STREAM_TRACKING: ParticleData's constructor
//! leaves seeds_ uninitialised and from_source() does not touch it, so a
//! partially seeded particle picks up stack garbage for the other streams.
//! STREAM_URR_PTABLE in particular is advanced on every energy change in
//! sample_neutron_reaction(), so leaving it unseeded silently decorrelates
//! the trees wherever a nuclide has unresolved resonances -- and a null
//! perturbation then returns noise instead of zero.
void run_one_tree(const BranchSite& site, int tree, int64_t seed_id)
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
  p.id() = seed_id;
  init_particle_seeds(seed_id, p.seeds());
  p.stream() = STREAM_TRACKING;

  score_site(tree, 1, site.wgt); // the root is this tree's depth-0 weight

  transport_history_based_single_particle(p);

  p.local_secondary_bank().clear();
}

} // namespace

//==============================================================================
// Input
//==============================================================================

void read_perturbations_xml()
{
  // Optional, like tallies.xml.
  std::string filename = settings::path_input + "perturbations.xml";
  if (!file_exists(filename))
    return;

  write_message("Reading perturbations XML file...", 5);

  pugi::xml_document doc;
  doc.load_file(filename.c_str());
  read_perturbations_xml(doc.document_element());
}

void read_perturbations_xml(pugi::xml_node root)
{
  int32_t next_id = 1;
  for (pugi::xml_node node : root.children("local_perturbation")) {
    settings::bep_on = true;
    Perturbation p;
    p.id = check_for_node(node, "id") ? std::stoi(get_node_value(node, "id"))
                                      : next_id;
    next_id = p.id + 1;

    // A perturbation is a SET of substitutions applied together, which is what
    // lets a displacement be expressed as the trailing sliver reverting and
    // the leading sliver taking the sample. The bare <cell>/<material> pair is
    // kept as shorthand for the one-cell case.
    for (pugi::xml_node node_s : node.children("substitution")) {
      Substitution s;
      s.cell_id = std::stoi(get_node_value(node_s, "cell"));
      s.mat_id = std::stoi(get_node_value(node_s, "material"));
      p.subs.push_back(s);
    }
    if (check_for_node(node, "cell")) {
      Substitution s;
      s.cell_id = std::stoi(get_node_value(node, "cell"));
      s.mat_id = std::stoi(get_node_value(node, "material"));
      p.subs.push_back(s);
    }
    perturbations.push_back(p);
  }
}

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
  if (perturbations.empty())
    fatal_error("<local_perturbation> given with no perturbations defined.");
  for (const auto& t : model::tallies) {
    if (t->adjoint_) {
      // Checked here, not in the Tally constructor: tallies.xml is read
      // before perturbations.xml, so settings::bep_on is still false while
      // tallies are being built and the guard there can never fire.
      fatal_error("<local_perturbation> cannot be combined with adjoint "
                  "(super-history) tallies: both drive the revival loop, with "
                  "different generation counts and different "
                  "global-contribution rules. Run them separately.");
    }
  }
  if (settings::ifp_on) {
    // ifp() indexes simulation::ifp_source_* by current_work(), which a
    // shadow particle does not own.
    fatal_error("<local_perturbation> cannot be combined with IFP kinetics "
                "parameters. Run them separately.");
  }

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

      if (s.mat_id == 0) {
        // Material id 0 means void, matching how an unfilled cell is stored.
        // Perturbing a sample to void is the natural "sample against nothing"
        // reference, so it has to be expressible.
        s.mat_index = MATERIAL_VOID;
      } else {
        auto m = model::material_map.find(s.mat_id);
        if (m == model::material_map.end()) {
          fatal_error(fmt::format("<local_perturbation> {}: material {} not "
                                  "found.",
            p.id, s.mat_id));
        }
        s.mat_index = m->second;
      }

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

  // NOTE: settings::super_n_generation is deliberately NOT touched here.
  // The driver reads it in event_check_limit_and_revive(), so changing it
  // would alter driver behaviour -- and BEP must leave the driver
  // bit-identical to a stock run. Shadow particles get their own revival
  // limit from bep_n_generation, keyed off bep_tree(). Same reasoning for
  // simulation::superhistory_on, which BEP no longer forces on:
  // create_fission_sites() tests settings::bep_on directly instead.

  int nd = settings::bep_n_generation + 1;
  size_t np = perturbations.size();
  tau.assign(trees.size() * nd, 0.0);
  thread_tau.assign(static_cast<size_t>(num_threads()) * tau_stride(), 0.0);
  thread_branch_sites.assign(num_threads(), {});
  tau_history.clear();
  branch_sites.clear();

  // One row per active generation. Warn rather than surprise the user with a
  // large allocation late in the run.
  double mb = 8.0 * settings::n_max_batches * settings::gen_per_batch *
              trees.size() * nd / (1024.0 * 1024.0);
  if (mb > 256.0) {
    warning(fmt::format("<local_perturbation> will record about {:.0f} MB of "
                        "per-generation data ({} trees, L = {}).",
      mb, trees.size(), settings::bep_n_generation));
  }

  size_t n_cells_touched = trees.size() - np;
  write_message(
    fmt::format("BEP: {} perturbation(s) over {} cell(s), {} shadow trees, "
                "L = {}.",
      np, n_cells_touched, trees.size(), settings::bep_n_generation),
    5);
}

void reset_generation()
{
  // Decide once per generation whether BEP does anything at all. Shadow trees
  // are pure overhead before the fission source has converged, and leaving
  // the flag clear keeps the branch test out of the transport hot path
  // entirely during inactive batches rather than testing and discarding on
  // every event.
  branching =
    settings::bep_on && simulation::current_batch > settings::n_inactive;
  if (!branching)
    return;
  std::fill(tau.begin(), tau.end(), 0.0);
  std::fill(thread_tau.begin(), thread_tau.end(), 0.0);
  for (auto& sites : thread_branch_sites)
    sites.clear();
  branch_sites.clear();
}

//==============================================================================
// Branch detection (driver side)
//==============================================================================

void maybe_branch(Particle& p, int32_t cell_index)
{
  // Caller has established that branching is on, that p is a trunk particle,
  // and that cell_index is touched by at least one perturbation.
  if (!p.type().is_neutron())
    return;
  if (p.wgt() == 0.0)
    return;

  BranchSite site;
  site.r = p.r();
  site.u = p.u();
  // SourceSite::E carries a GROUP INDEX in multigroup mode, not an energy:
  // from_source() does g() = int(src->E) there. Storing p.E() unconditionally
  // fed an energy in as a group index, which runs off the end of every
  // group-indexed array. Same idiom as create_secondary() and split().
  site.E = settings::run_CE ? p.E() : p.g();
  site.wgt = p.wgt();
  site.time = p.time();
  site.cell = cell_index;
  // Identity of the branch, not its arrival order. (id, n_tracks, n_event)
  // is unique for a driver particle within a generation -- n_event alone is
  // not, because event_revive_from_secondary() resets it.
  site.seed_id = branch_seed(p.id(), p.n_tracks(), p.n_event());

  // No synchronisation: this runs in the transport loop, and an omp critical
  // here serialises every thread on a push_back. The vectors are merged in
  // run_shadow_pass(), where the order no longer matters because each site
  // carries its own seed.
  thread_branch_sites[thread_num()].push_back(site);

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
  // The innermost operation of the whole feature, once per shadow fission
  // site. Writing to a per-thread slab rather than an atomic add on shared
  // memory: the shared array is a few hundred bytes, so every thread would
  // contend for the same handful of cache lines billions of times.
  thread_tau[static_cast<size_t>(thread_num()) * tau_stride() +
             tau_index(tree, depth)] += wgt;
}

//==============================================================================
// Shadow pass
//==============================================================================

void run_shadow_pass()
{
  if (!branching)
    return;

  // Merge the per-thread vectors and put them in a deterministic order. The
  // order the threads recorded them in varies run to run; sorting by the
  // seed, which is a property of the branch itself, makes the whole shadow
  // pass -- including the order the per-thread tau slabs are summed in --
  // reproducible for a given thread count.
  branch_sites.clear();
  for (const auto& sites : thread_branch_sites)
    branch_sites.insert(branch_sites.end(), sites.begin(), sites.end());
  if (branch_sites.empty())
    return;
  std::sort(branch_sites.begin(), branch_sites.end(),
    [](const BranchSite& a, const BranchSite& b) {
      return a.seed_id < b.seed_id;
    });
  for (const auto& site : branch_sites) {
    ++n_branch_total;
    w_branch_total += site.wgt;
  }

  auto n = static_cast<int64_t>(branch_sites.size());

  // Static, not dynamic: with the sites in a fixed order it gives each thread
  // a fixed chunk, so the per-thread partial sums are reproducible. Tree cost
  // varies a lot -- it is a branching process -- but with many sites per
  // thread that averages out. Switch to dynamic if load imbalance ever shows
  // up, at the cost of bit-reproducibility.
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    const BranchSite& site = branch_sites[i];

    // One seed per branch site, shared by the reference tree and every
    // perturbed tree spawned here, so all of them draw the same numbers
    // wherever they are doing the same physics. That is what collapses the
    // variance of a difference, and what makes a null perturbation return
    // zero rather than noise. It was fixed when the site was recorded, so it
    // does not depend on this loop's index -- see BranchSite::seed_id.
    int64_t seed_id = site.seed_id;

    run_one_tree(site, cell_ref_tree[site.cell], seed_id);

    // Only the perturbations that touch THIS cell spawn a tree here. A
    // perturbation whose other cells the history later reaches is picked up
    // by a separate branch there, since the driver stays untagged.
    for (int ip : cell_perts[site.cell]) {
      run_one_tree(site, perturbations[ip].tree, seed_id);
    }
  }
}

void accumulate_generation()
{
  if (!branching)
    return;

  // Sum the per-thread slabs, in thread order.
  std::fill(tau.begin(), tau.end(), 0.0);
  int stride = tau_stride();
  for (int t = 0; t < num_threads(); ++t) {
    for (int i = 0; i < stride; ++i)
      tau[i] += thread_tau[static_cast<size_t>(t) * stride + i];
  }

#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    // Sum tau across ranks before recording: the estimator is a ratio of sums
    // over every progenitor, not a mean of per-rank ratios.
    vector<double> reduced(tau.size(), 0.0);
    MPI_Allreduce(tau.data(), reduced.data(), static_cast<int>(tau.size()),
      MPI_DOUBLE, MPI_SUM, mpi::intracomm);
    tau = reduced;
  }
#endif

  // Record and nothing else. Deliberately no ratio, no logarithm and no
  // per-generation realisation: a shadow tree is a branching process that
  // can go extinct, so tau for a whole generation can be zero over a small
  // number of branch sites, and log(0) is -inf. Ordinary IFP estimators are
  // robust to this precisely because they sum over every progenitor before
  // dividing, and BEP now does the same. Dropping degenerate generations
  // would bias the worth, since extinction correlates with the strength of
  // the perturbation.
  tau_history.insert(tau_history.end(), tau.begin(), tau.end());
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
  write_dataset(group, "n_generations_recorded", n_generations);
  write_dataset(group, "n_trees", static_cast<int>(trees.size()));
  write_dataset(group, "n_branch", n_branch_total);
  write_dataset(group, "w_branch", w_branch_total);
  write_dataset(group, "n_perturbations", np);

  vector<int32_t> ids;
  for (const auto& p : perturbations)
    ids.push_back(p.id);
  write_dataset(group, "ids", ids);

  // Flat [generation][tree][depth]; Python reshapes with n_trees and
  // n_generation. Raw, because forming the ratio, fitting the slope and
  // blocking for an uncertainty all belong where they can be changed without
  // a rebuild.
  write_dataset(group, "tau", tau_history);

  for (int ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];
    hid_t pg = create_group(group, fmt::format("perturbation {}", p.id));
    write_dataset(pg, "index", ip);
    write_dataset(pg, "tree", p.tree);

    // Which reference trees make up this perturbation's matched denominator:
    // exactly the cells it touches, i.e. exactly the branch sites where it
    // spawned a tree.
    vector<int32_t> ref_trees;
    for (int32_t ci : p.cells)
      ref_trees.push_back(cell_ref_tree[ci]);
    write_dataset(pg, "ref_trees", ref_trees);

    vector<int32_t> cids, mids;
    for (const auto& sub : p.subs) {
      cids.push_back(sub.cell_id);
      mids.push_back(sub.mat_id);
    }
    write_dataset(pg, "cells", cids);
    write_dataset(pg, "materials", mids);

    close_group(pg);
  }

  close_group(group);
}

} // namespace bep
} // namespace openmc
