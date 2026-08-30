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
//! tests settings::bep_on directly for the local bank). Shadow particles also
//! own no slot in any per-source array, so event_death() skips the
//! progeny_per_particle write for them. Between them these keep the driver
//! bit-identical to a stock run, which the fission source in
//! test_driver_is_unperturbed asserts exactly.

#include <cstdint>

#include <pugixml.hpp>

#include "openmc/hdf5_interface.h"
#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {

class Particle;

constexpr int BEP_TRUNK {-1}; //!< value of bep_tree() for a driver particle

namespace settings {

extern bool bep_on;
//! Both declared in settings.h and parsed from settings.xml -- L is a
//! property of the run, not of any one perturbation, so it belongs with
//! superhistory_n_generation rather than in perturbations.xml.
extern int bep_n_generation;
extern int super_n_generation;

} // namespace settings

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

//! A shadow population. `pert` is an index into `perturbations`, or -1 for a
//! reference tree.
struct Tree {
  int pert;
};

extern vector<Perturbation> perturbations;
extern vector<Tree> trees;

//! cell index -> reference tree index, or -1 if untouched. Sized to
//! model::cells so the hot-path test is one indexed load.
extern vector<int> cell_ref_tree;

//! cell index -> perturbations touching that cell.
extern vector<vector<int>> cell_perts;

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

//! True only while a generation that BEP acts on is being processed: set at
//! the start of each generation and left set through the shadow pass in
//! finalize_generation(). Every BEP hot-path test reads this rather than
//! settings::bep_on, so the feature is genuinely inert during inactive
//! batches instead of merely declining to record.
extern bool branching;

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
  return static_cast<int>(trees.size()) * (settings::bep_n_generation + 1);
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

//! Read perturbations.xml if present. Optional, like tallies.xml.
void read_perturbations_xml();

//! Read a <perturbations> element, from perturbations.xml or from model.xml.
void read_perturbations_xml(pugi::xml_node root);

void init();
void reset_generation();
void maybe_branch(Particle& p, int32_t cell_index);
void score_site(int tree, int super_gen, double wgt);
void run_shadow_pass();
void accumulate_generation();
void write_results(hid_t file_id);

} // namespace bep
} // namespace openmc

#endif // OPENMC_BEP_H
