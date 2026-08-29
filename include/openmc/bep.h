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
//! and rho is the SLOPE of l_p(d). The intercept is the mode-overlap
//! difference; keeping it out of the slope is what makes this exact rather
//! than first order. Note the LOG: the ratio of importance sums appears as a
//! multiplicative constant, so it cancels from the slope exactly. Fitting the
//! bare ratio r = tau_p/R_p - 1 instead scales the slope by that constant,
//! which is second order but not negligible when a sample strongly depresses
//! the local importance. l is formed as log1p(r) with r built from the
//! correlated difference, so the cancellation survives the log.
//!
//! The per-generation realisation accumulated is the finite difference over
//! the upper half of the depth range,
//!
//!     rho_gen = [l_p(L) - l_p(L/2)] / (L - L/2)
//!
//! Cross products rho_gen(i) * rho_gen(j) are accumulated for every pair, so
//! any linear combination of perturbations -- a difference between two sample
//! positions, a finite-difference derivative, a fitted traverse -- gets an
//! exact uncertainty. Perturbations sharing branch sites and seeds are
//! strongly correlated, so those combinations are far better determined than
//! the individual worths. That is the point of running them together.
//!
//! Sign: rho < 0 for an added absorber.
//!
//! DEPTH ENCODING
//! --------------
//! Shadow roots launch at super_gen == 1, not 0, so every existing
//! `super_gen() <= 0` gate in the tree excludes shadow particles
//! automatically. Relative depth is d = super_gen - 1, and
//! settings::super_n_generation is set to L + 1.

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
extern int bep_n_generation; //!< L, shared by all perturbations

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
};

//! True only while a generation that BEP acts on is being processed: set at
//! the start of each generation and left set through the shadow pass in
//! finalize_generation(). Every BEP hot-path test reads this rather than
//! settings::bep_on, so the feature is genuinely inert during inactive
//! batches instead of merely declining to record.
extern bool branching;

extern vector<BranchSite> branch_sites;
extern vector<double> tau;     //!< [tree * (L + 1) + depth], this generation
extern vector<double> sum_l;   //!< [pert * (L + 1) + depth], diagnostics
extern vector<double> sum_tau; //!< [tree * (L + 1) + depth], diagnostics
extern vector<double> sum_rho;
extern vector<double> sum_cross; //!< [i * n_pert + j]
extern vector<int64_t> n_pair;   //!< [i * n_pert + j]
extern int64_t n_generations;
extern int64_t n_branch_total;
extern double w_branch_total;

inline int tau_index(int tree, int depth)
{
  return tree * (settings::bep_n_generation + 1) + depth;
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

//! Material index perturbation `pert` substitutes into `cell_index`, or -1 if
//! it does not touch that cell. Substitution lists are tiny, so a linear scan
//! beats any map here.
inline int32_t substitute(int pert, int32_t cell_index)
{
  for (const auto& s : perturbations[pert].subs) {
    if (s.cell_index == cell_index)
      return s.mat_index;
  }
  return -1;
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
