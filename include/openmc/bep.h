#ifndef OPENMC_BEP_H
#define OPENMC_BEP_H

//! \file bep.h
//! Branched Exact Perturbation (BEP).
//!
//! Reactivity worth of one or more local perturbations, from a single
//! eigenvalue run. A perturbation is a SET of cell->material substitutions
//! applied together, so it covers a sample swap (one cell), a sample
//! DISPLACEMENT (two thin slivers, one losing the sample and one gaining it),
//! and multi-region changes such as a voided plenum. Exposed to users as
//! repeated <local_perturbation> elements in settings.xml.
//!
//! LINEAGE
//! -------
//! Exact perturbation theory in Monte Carlo transport is due to Truchet et al.
//! (SNA+MC 2013; PHYSOR 2014) and their Black Body Exact Perturbation for
//! small samples (M&C 2019). BEP keeps their formulation and their IFP
//! importance estimator but replaces the black-body intermediate state with
//! LAZY BRANCHING at the perturbation boundary.
//!
//! For displacement specifically: a rigid translation is NOT a virtual-density
//! problem (that theory handles uniform expansion by stretching mean free
//! paths). It is exactly a pair of material substitutions on the symmetric
//! difference of the two positions, which is why the multi-substitution form
//! below covers it with no new physics. Compare Burke & Kiedrowski, NSE
//! 189(3) 199-223 (2018), which gets dk/d(dimension) from an adjoint-weighted
//! surface term and needs a kernel-density bandwidth to estimate the surface
//! flux; here the sliver thickness plays that role but is a physical length
//! you can converge rather than a bandwidth you must tune.
//!
//! BEP has NOT been validated against the published BBEP results. Reproduce
//! the M&C 2019 7x7 doped-pin cases before trusting it (see BEP_README.md).
//!
//! ESTIMATOR
//! ---------
//! The driver is an ordinary eigenvalue calculation in the REFERENCE state.
//! It is never modified, so k, the fission source and all ordinary tallies
//! stay bit-identical to a stock run.
//!
//! When a driver particle enters any cell touched by any perturbation, its
//! phase point is recorded as a BRANCH SITE. After the generation finishes
//! each site is propagated privately for L generations: once in that cell's
//! REFERENCE tree, and once per perturbation that touches that cell. A
//! perturbation's tree applies ALL of its own substitutions and none of any
//! other's, so trees interact correctly with regions they do not own.
//!
//! Let tau_t(d) be the depth-d descendant weight in tree t, and for a
//! perturbation p let R_p(d) be the sum of the reference trees over exactly
//! the cells p touches -- i.e. over exactly the branch sites where p spawned
//! a tree. Asymptotically tau -> W * phi^dagger * k^d, so
//!
//!     l_p(d) = ln[ tau_p(d) / R_p(d) ] = const + d * ln(k_p / k_ref)
//!
//! and the SLOPE of l_p(d) is ln(k_p / k_ref), i.e. dk/k. That is NOT a
//! reactivity: the reactivity difference is 1/k_ref - 1/k_p, smaller by a
//! factor of k. Python converts it (Perturbations._set_results); this file
//! records only tau, so nothing here needs k. The intercept is the
//! mode-overlap difference; keeping it out of the slope is what makes this
//! exact rather than first order. Note the LOG: the ratio of importance sums
//! appears as a multiplicative constant, so it cancels from the slope
//! exactly. Fitting the
//! bare ratio r = tau_p/R_p - 1 instead scales the slope by that constant,
//! which is second order but not negligible when a sample strongly depresses
//! the local importance. l is formed as log1p(r) with r built from the
//! correlated difference, so the cancellation survives the log.
//!
//! CRITICALLY, the ratio is formed ONCE, from sums accumulated over the whole
//! run -- never per generation. A shadow tree is a branching process and can
//! go extinct: with mean offspring k the extinction probability of a single
//! tree is the root of q = exp(k(q-1)), about 0.16 even at k = 2.2. Over a
//! handful of branch sites the whole population can die, and log(0) is -inf.
//! Summing over every progenitor first is what makes ordinary IFP estimators
//! robust to this -- a dead tree simply contributes zero -- and BEP does the
//! same. Dropping degenerate generations instead would have been worse than
//! noisy: extinction correlates with the perturbation's strength, so the
//! selection biases the worth.
//!
//! So this file records only tau_t(d) per generation. Forming l_p(d), fitting
//! the slope, and blocking the generations for an uncertainty all happen in
//! Python, where the block size can be adapted to the branch rate without a
//! rebuild. Blocks are also what give the covariance between perturbations,
//! and hence exact uncertainties on any linear combination -- a difference
//! between two sample positions, a finite-difference derivative, a fitted
//! traverse.
//!
//! Sign: rho < 0 for an added absorber.
//!
//! DEPTH ENCODING, AND DRIVER ISOLATION
//! ------------------------------------
//! Shadow roots launch at super_gen == 1, not 0, so every existing
//! `super_gen() <= 0` gate in the tree excludes shadow particles
//! automatically. Relative depth is d = super_gen - 1.
//!
//! BEP mutates NO global that the driver reads. It does not set
//! settings::super_n_generation (event_check_limit_and_revive() derives a
//! shadow tree's limit from bep_n_generation via bep_tree() instead) and it
//! does not force simulation::superhistory_on on (create_fission_sites()
//! tests simulation::bep_on directly for the local bank). Shadow particles
//! own no slot in any per-source array, so event_death() skips the
//! progeny_per_particle write for them. Between them these keep the driver
//! bit-identical to a stock run, which the fission source in
//! test_driver_is_unperturbed asserts exactly.

#include <cstdint>

#include <pugixml.hpp>

#include "openmc/hdf5_interface.h"
#include "openmc/position.h"
// For settings::bep_n_generation and settings::super_n_generation, both used
// by the inline functions below. Re-declaring them here instead would be two
// declarations of one variable to keep in sync by hand, and a type that
// drifted between them is an ODR violation the linker does not diagnose.
#include "openmc/settings.h"
#include "openmc/vector.h"

namespace openmc {

class Particle;

constexpr int BEP_TRUNK {-1}; //!< value of bep_tree() for a driver particle

//! Floor on tree_site_weight, so that a tree whose weight is measured
//! anomalously small one generation cannot ask for an unbounded population
//! the next.
constexpr double MIN_SITE_WEIGHT {1.0e-8};

//! Generations of tau a tree must carry at one site weight before its spread
//! is trusted to set the next one. Below this the rule uses its bootstrap.
constexpr int64_t MIN_STAT_GENERATIONS {5};

// Whether BEP is configured at all is `!bep::perturbations.empty()`. There
// is no separate flag: two representations of one fact have to be kept in
// step, and the vector is the one that carries the information.
//
// Whether BEP is running THIS batch is simulation::bep_on, declared in
// simulation.h beside superhistory_on and set by reset_generation().

namespace bep {

//! One cell->material replacement.
struct Substitution {
  int32_t cell_id;
  int32_t mat_id;
  int32_t cell_index {-1};
  int32_t mat_index {-1};
};

//! A set of substitutions applied together. For a displacement this is the
//! trailing sliver reverting to the displaced material and the leading sliver
//! taking the sample.
struct Perturbation {
  int32_t id;
  vector<Substitution> subs;
  vector<int32_t> cells; //!< cell indices touched, for the matched reference
  int tree {-1};
};

//! Value in `tree_pert` for a reference tree, which belongs to no
//! perturbation. Named because the bare -1 was previously written three
//! different ways in three unrelated sentinels.
constexpr int BEP_NO_PERT {-1};

extern vector<Perturbation> perturbations;

//! Shadow tree index -> index into `perturbations`, or BEP_NO_PERT for a
//! reference tree. One entry per tree, so `tree_pert.size()` is the number
//! of shadow populations and the first dimension of `tau`.
extern vector<int> tree_pert;

//! cell index -> reference tree index, or -1 if untouched. Sized to
//! model::cells so the hot-path test is one indexed load.
extern vector<int> cell_ref_tree;

//! cell index -> perturbations touching that cell.
extern vector<vector<int>> cell_perts;

//! Weight each tree banks its shadow fission sites at, one per tree.
//!
//! An ordinary eigenvalue calculation banks fission sites of UNIT weight and
//! puts the parent's weight into the PROBABILITY of banking one at all --
//! see create_fission_sites(), where nu is an integer count. That is fine
//! while every neutron weighs about one, and ruinous below it: a neutron of
//! weight 1e-3 becomes a one-in-a-thousand lottery for a weight-1 site, so
//! everything a low-weight source gains by being sampled smoothly is thrown
//! away again at its first fission.
//!
//! A tree's sites are banked at this weight instead, and their number scales
//! up to match, which leaves the banked weight unchanged and its variance far
//! lower. The value is a weight window on the shadow fission bank: 1.0 is
//! exactly what an eigenvalue calculation does today, and is what every tree
//! carrying full-weight neutrons keeps.
extern vector<double> tree_site_weight;

//! Weight tree `tree` banks its fission sites at. Always 1 for a driver
//! particle, so nothing outside a shadow tree is affected.
inline double site_weight(int tree)
{
  return (tree >= 0 && tree < static_cast<int>(tree_site_weight.size()))
           ? tree_site_weight[tree]
           : 1.0;
}

//! Weight the shadow tree currently being transported on this thread was
//! rooted at, i.e. the weight of the branch site it grew from, or 0 outside a
//! shadow tree.
//!
//! This is the natural scale for anything inside a shadow tree: every weight
//! in the tree is some fraction of it. Expressing a cutoff against it makes
//! that cutoff invariant to how the driver happens to normalise its weights
//! and to the population the site weight targets, which an absolute cutoff
//! would not
//! be. Per thread because run_one_tree() transports one tree at a time on
//! each thread, and 0 outside one so that a driver particle can never be
//! affected by a setting meant for a shadow tree.
double root_weight();

//! Does `tree` belong to a perturbation, as opposed to being a reference tree
//! or the trunk?
//!
//! What separates a tree whose population is the perturbation's own from one
//! that is merely the reference it is scored against. A reference tree
//! carries the same full-weight population an ordinary eigenvalue calculation
//! would, so variance reduction aimed at a perturbation's thin, low-weight
//! population has no business touching it.
inline bool in_perturbation_tree(int tree)
{
  return tree >= 0 && tree < static_cast<int>(tree_pert.size()) &&
         tree_pert[tree] >= 0;
}

//! Independent source events that seeded each tree this generation: the
//! branch sites a material perturbation's tree grew from, the photoneutron
//! births a photonuclear one's did. Counted, never estimated.
//!
//! This is the M of the variance law, relative variance = 1/n + c/M. Nothing
//! splitting does can raise it -- it is how many genuinely independent
//! samples the physics offered -- which is why it sets the floor the
//! population is optimised against.
extern vector<int64_t> tree_sources;

//! Weight a typical particle in `tree` carries, relative to the reference
//! tree it is scored against. Measured, and independent of every
//! variance-reduction knob.
//!
//! Distinct from site_weight() on purpose. That one is chosen to hit a
//! target POPULATION, so it moves with whatever the rule decides that
//! population should be. This one is just w_pert / w_ref: what the
//! perturbation's
//! population actually weighs, which for a photonuclear perturbation is the
//! photonuclear production ratio aggregated over the problem (measured
//! 2.3e-4, against a per-collision neutron_prod/total * yield of 1.4e-4 --
//! the difference being the photoneutrons' own fission progeny).
//!
//! Keeping them apart matters: anything that scales a cutoff by site_weight()
//! moves that cutoff whenever the population ratio is tuned, silently
//! coupling two controls that have to stay independent.
extern vector<double> tree_weight_scale;

inline double weight_scale(int tree)
{
  return (tree >= 0 && tree < static_cast<int>(tree_weight_scale.size()))
           ? tree_weight_scale[tree]
           : 1.0;
}

//! The weight a typical particle in `tree` is born at.
//!
//! This is the scale a shadow tree's variance reduction has to measure
//! against, and it is NOT the tree's root weight: a perturbation's population
//! can sit decades below the branch site its tree grew from. Judging such a
//! particle against the root weight declares the whole tree negligible;
//! judging it against this says what was meant.
//!
//! Exactly the root weight for a reference tree and for a material
//! perturbation's tree, both of which carry a full-weight population and have
//! weight_scale 1.0, so neither is disturbed by anything keyed off this.
inline double characteristic_weight(int tree)
{
  return root_weight() * weight_scale(tree);
}

//! Choose each tree's site weight from the weight it actually carried this
//! generation, so that every tree transports a comparable number of sites
//! whatever weight its particles happen to have. Called once per generation.
void update_site_weights();

struct BranchSite {
  Position r;
  Direction u;
  double E;
  double wgt;
  double time;
  int32_t cell;

  //! Seed for every shadow tree spawned here.
  //!
  //! Derived from the driver particle's own identity, NOT from this site's
  //! position in `branch_sites`: that vector is filled under an omp critical
  //! from a schedule(runtime) loop, so its order depends on thread timing.
  //! Indexing by it gave a different seed to the same physical branch point
  //! on every run, and BEP results were not reproducible.
  int64_t seed_id;
};

//! Branch sites collected during the current generation, one vector per
//! thread. Per-thread rather than shared because the alternative is an omp
//! critical inside the transport loop, which serialises every thread on a
//! push_back. Merged and sorted at the start of the shadow pass.
extern vector<vector<BranchSite>> thread_branch_sites;

//! The merged, sorted branch sites the shadow pass iterates over.
extern vector<BranchSite> branch_sites;

//! Depth-d descendant weight for this generation, one slab per thread,
//! indexed [thread][tree][depth].
//!
//! Per-thread for the same reason, only more so: the shared version needed an
//! omp atomic per shadow fission site, which is the innermost loop of the
//! whole feature -- billions of atomic adds onto a handful of cache lines,
//! contended by every thread. The slabs cost a few kB and are summed once per
//! generation.
extern vector<double> thread_tau;

//! Per-thread slabs summed, i.e. this generation's tau. [tree * (L+1) + depth]
extern vector<double> tau;

//! Per-generation record of `tau`, appended once per active generation and
//! laid out as [generation][tree][depth]. Everything downstream is derived
//! from this in Python. Costs n_generations * n_trees * (L + 1) doubles.
extern vector<double> tau_history;
extern int64_t n_generations;
extern int64_t n_branch_total;
extern double w_branch_total;

//! Offset within one thread's slab, and within the merged `tau`.
inline int tau_index(int tree, int depth)
{
  return tree * (settings::bep_n_generation + 1) + depth;
}

//! Number of doubles in one thread's slab.
inline int tau_stride()
{
  return static_cast<int>(tree_pert.size()) * (settings::bep_n_generation + 1);
}

//! Reference tree owning `cell_index`, or -1 if no perturbation touches it.
//! Safe before init().
inline int ref_tree_of_cell(int32_t cell_index)
{
  return (cell_index >= 0 &&
           cell_index < static_cast<int32_t>(cell_ref_tree.size()))
           ? cell_ref_tree[cell_index]
           : -1;
}

//! Material perturbation `pert` substitutes into `cell_index`. Returns false
//! if it does not touch that cell. Out-param rather than a sentinel return,
//! because MATERIAL_VOID is itself -1. Substitution lists are tiny, so a
//! linear scan beats any map here.
inline bool substitute(int pert, int32_t cell_index, int32_t& mat_index)
{
  for (const auto& s : perturbations[pert].subs) {
    if (s.cell_index == cell_index) {
      mat_index = s.mat_index;
      return true;
    }
  }
  return false;
}

//! How deep this particle's super-history chain runs.
//!
//! Keyed off the particle's tree so that BEP never has to overwrite
//! settings::super_n_generation, which the driver and the adjoint
//! super-history both read. Every gate that bounds a chain -- fission-site
//! creation in sample_neutron_reaction() and revival in
//! event_check_limit_and_revive() -- must use THIS, or shadow trees silently
//! stop growing and every tau beyond depth 0 comes out zero.
inline int generation_limit(int bep_tree_value)
{
  return (bep_tree_value != BEP_TRUNK) ? settings::bep_n_generation + 1
                                       : settings::super_n_generation;
}

//! Give every substitution target the temperatures of the cells it can be
//! substituted into.
//!
//! Multigroup builds macroscopic data only for materials that appear in a
//! CELL (see MgxsInterface::get_mat_kTs), and a perturbation's target need
//! not appear anywhere in the geometry -- that is the normal case, since it
//! is the material being swapped IN. Without this the target gets a blank
//! Mgxs and the first shadow particle to reach it indexes an empty
//! temperature vector, which segfaults rather than erroring.
//!
//! Called from get_mat_kTs(), which is why perturbations.xml is read before
//! the cross sections are finalized.
void add_substitution_temperatures(vector<vector<double>>& kTs);

//! The same, one level down: the NUCLIDE temperatures to read.
//!
//! get_temperatures() also walks cells only, so a substitution target's
//! nuclides are read with an empty temperature list and the macroscopic
//! build then has nothing to interpolate. Both hooks are needed -- giving
//! the material a temperature without giving its nuclides one just moves
//! the crash from Mgxs::calculate_xs() into the Mgxs constructor.
//!
//! NOTE the units differ from add_substitution_temperatures(): this list
//! holds temperature in K, that one holds kT in eV.
void add_substitution_nuclide_temperatures(
  vector<vector<double>>& nuc_temps);

void init();
void reset_generation();
void maybe_branch(Particle& p, int32_t cell_index);
void score_site(int tree, int super_gen, double wgt);
void run_shadow_pass();
void accumulate_generation();
void write_results(hid_t file_id);

} // namespace bep

//==============================================================================
// Functions
//==============================================================================

// In namespace openmc, not bep: every other input reader is
// (read_settings_xml, read_materials_xml, read_geometry_xml ...), including
// read_settings_xml, whose data lives in namespace settings exactly as this
// reader's data lives in namespace bep. The namespace is for the data, not
// for the function that fills it.

//! Read perturbations.xml if present. Optional, like tallies.xml.
void read_perturbations_xml();

//! Read a <perturbations> element, from perturbations.xml or from model.xml.
void read_perturbations_xml(pugi::xml_node root);
} // namespace openmc

#endif // OPENMC_BEP_H
