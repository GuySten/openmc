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
//! the M&C 2019 7x7 doped-pin cases before trusting it. What HAS been checked
//! is in docs/bep_level_design.md: a null substitution returns exactly zero, a
//! displacement across a plane of symmetry returns zero, and a B10 sample
//! worth -265 +/- 50 pcm against -263 +/- 5 from a direct k'-k difference.
//!
//! ESTIMATOR
//! ---------
//! The driver is an ordinary eigenvalue calculation in the REFERENCE state.
//! It is never modified, so k, the fission source and all ordinary tallies
//! stay bit-identical to a stock run.
//!
//! BEP is a LEVEL estimator, read at one depth. Per perturbation it grows five
//! SOURCE-ROOTED populations, all in that perturbation's perturbed physics:
//!
//!   D   the reference fission source, sampled from simulation::fission_bank,
//!       i.e. (1/k) F psi -- the denominator;
//!   F+  the fission-production source chi' nu Sigma_f' psi,
//!   F-  the fission-production source chi  nu Sigma_f  psi,
//!   L+  the positive part of -(dL psi) = -(dSigma_t psi) + (dSigma_s psi),
//!   L-  its negative part, carried as a POSITIVE population.
//!
//! Grown d generations in the perturbed physics, each population's total
//! weight tends to k'^d times its overlap with the PERTURBED adjoint,
//! <phi'^dag, .>. That common k'^d cancels in the combination in level()
//! (see below), which is therefore FLAT in d -- a level, not a slope. The
//! worth is read at d = L; the whole curve d = 0..L is the convergence
//! diagnostic, and there is no fit window to choose and no window bias to
//! correct. Signs are carried by which population a root feeds, never by a
//! negative weight: the +/- pairs are subtracted only at scalar readout.
//!
//! This is the perturbed-adjoint (exact) form of perturbation theory. The
//! importance that weights the source is built by the propagation AFTER the
//! perturbation, so propagating for the full depth in the PERTURBED physics is
//! what makes phi'^dag perturbed and the result exact rather than first order.
//!
//! The ratio is formed ONCE per BATCH, from sums over that batch's
//! generations, never per generation. A shadow tree is a branching process and
//! can go extinct; summing over every root before dividing is what makes
//! ordinary IFP estimators robust to that, and BEP does the same. The
//! uncertainty is then the ordinary spread of the per-batch realisations over
//! active batches -- standard OpenMC batch statistics, with
//! generations_per_batch chosen so batches are effectively independent. No
//! jackknife, no log, no slope fit, no correlation model anywhere.
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

//! Floor on tree_site_weight, so that a population whose weight is measured
//! anomalously small one generation cannot ask for an unbounded number of
//! sites the next.
constexpr double MIN_SITE_WEIGHT {1.0e-8};

//! Fewest banked sites a tuned population may be driven to.
//!
//! The optimum (docs/bep_autotune.md 7') can ask for a very small N when a
//! population carries little of the answer -- the denominator above all, now
//! that it is tunable. Below a few dozen sites per batch the population
//! estimates nothing and the variance model behind the optimum stops meaning
//! anything, so the site COUNT is bounded rather than the weight: that is
//! what is actually required, and it leaves the weight free to exceed one.
constexpr double MIN_TREE_SITES {100.0};

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
  vector<int32_t> cells; //!< cell indices touched (where the source is emitted)

  //! The five SOURCE-ROOTED shadow-tree classes, all transported in this
  //! perturbation's PERTURBED physics (all map to this pert in `tree_pert`):
  //!
  //!   tree_d  -- the reference fission source (1/k) F psi: the DENOMINATOR.
  //!   tree_fp -- the fission-production source  chi' nu Sigma_f' psi
  //!   tree_fn -- the fission-production source  chi  nu Sigma_f  psi
  //!   tree_lp -- the positive part of -(dL psi), the removal/scatter source
  //!   tree_ln -- its negative part, tracked as a POSITIVE population and
  //!              subtracted at readout, so no weight is ever negative.
  //!
  //! Why the fission channel is separate from the removal channel rather than
  //! summed into one signed source: the exact relation between the two is
  //!
  //!     N_F / k'  +  N_L  =  (1/k - 1/k') * <phi'^dag, F psi>
  //!
  //! -- the fission term carries a 1/k' the removal term does not. Folding
  //! them together before k' is known forces either a linearisation or an
  //! iteration; kept apart, the depth-d level inverts in closed form with no
  //! approximation at all (see level()). That matters because this estimator's
  //! whole claim is exactness at large worth.
  int tree_d {-1};
  int tree_fp {-1};
  int tree_fn {-1};
  int tree_lp {-1};
  int tree_ln {-1};
};

extern vector<Perturbation> perturbations;

//! Shadow tree index -> index into `perturbations`. Every tree now belongs to
//! a perturbation (there are no reference trees): each perturbation owns five
//! source-rooted trees, all transported in its perturbed physics. One entry
//! per tree, so `tree_pert.size()` is the number of shadow populations and the
//! first dimension of `tau`.
extern vector<int> tree_pert;

//! Shadow tree index -> its class; see Perturbation above.
enum TreeClass {
  TREE_D = 0,
  TREE_FP = 1,
  TREE_FN = 2,
  TREE_LP = 3,
  TREE_LN = 4,
  N_TREE_CLASS = 5
};
extern vector<int> tree_class;

//! cell index -> perturbations touching that cell (where the source is
//! emitted). Sized to model::cells so the hot-path test is one indexed load.
extern vector<vector<int>> cell_perts;

//! Is any perturbation's source emitted in this cell? The driver-side gate.
inline bool cell_touched(int32_t cell_index)
{
  return cell_index >= 0 &&
         cell_index < static_cast<int32_t>(cell_perts.size()) &&
         !cell_perts[cell_index].empty();
}

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
//! rooted at, i.e. the weight of the source root it grew from, or 0 outside a
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

//! Is `tree` one of the thin source trees (N+/N-), as opposed to the
//! full-weight denominator (D)? Variance reduction aimed at the perturbation's
//! thin, low-weight source population has no business touching D, which carries
//! the same full-weight population an ordinary eigenvalue calculation would.
inline bool is_source_tree(int tree)
{
  return tree >= 0 && tree < static_cast<int>(tree_class.size()) &&
         tree_class[tree] != TREE_D;
}

//! Independent source events that seeded each population this generation:
//! the roots it grew from -- fission-bank sites for the denominator, emitted
//! source particles for the rest. Counted, never estimated.
//!
//! This is the M of the variance law, relative variance = 1/n + c/M. Nothing
//! splitting does can raise it -- it is how many genuinely independent
//! samples the physics offered -- which is why it sets the floor the
//! population is optimised against.
extern vector<int64_t> tree_sources;

//! Per tree, summed over the batch: the number of roots, and the sum of the
//! SQUARES of each root's own depth-L descendant weight.
//!
//! Together with the population itself (which is the sum of those weights)
//! these give the per-source relative variance `c_t` of equation (1)
//! directly, rather than by subtracting it out of the measured spread:
//!
//!     c_t = M * sum(x^2) / (sum x)^2  -  1
//!
//! The roots ARE independent samples of exactly the quantity `c_t` describes,
//! and there are thousands of them every generation, so nothing has to be
//! inferred. That is what makes `perturbation_n_roots` settable from the
//! derivation instead of by the user (docs/bep_autotune.md 4b).
extern vector<int64_t> batch_sources;
extern vector<double> batch_root_sq;
extern vector<double> thread_root_sq;

//! The root count the rule chose, used when perturbation_n_roots is 0.
extern int64_t n_roots_auto;

//! Histories transported by the shadow pass over the whole run.
//!
//! The reproducible cost. A figure of merit needs a denominator, and wall
//! clock is not one: two runs of a single seed on a loaded machine give
//! different seconds and identical work. This is also the quantity the tuning
//! rule optimises against, so a measured figure of merit and the rule's own
//! objective are expressed in the same units.
extern int64_t n_history_total;

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

//! Choose each tree's site weight from the weight it actually carried this
//! generation, so that every tree transports a comparable number of sites
//! whatever weight its particles happen to have. Called once per generation.
void update_site_weights();

//! One driver track segment inside a cell some perturbation touches.
//!
//! The perturbation source dH psi is estimated with a TRACK-LENGTH estimator,
//! not a collision estimator: a collision estimator has no collisions to score
//! at where the reference material is void, so it cannot see a perturbation
//! that FILLS a void -- exactly the voided-plenum case run backwards. The
//! track-length form costs nothing extra and is lower variance besides.
//!
//! Everything the source needs is captured here, so the driver itself is only
//! ever READ -- never stopped, tagged, or re-seeded. The reference macroscopic
//! cross sections come along for free because the driver has just computed
//! them; only the SUBSTITUTED material's have to be evaluated, and that is
//! done in the shadow pass on a scratch particle, off the transport hot path.
struct TrackSite {
  Position r;        //!< START of the segment (before the flight)
  Direction u;
  double E;          //!< energy in CE, group index in MG
  double wgt;
  double time;
  double distance;   //!< length of the segment inside this cell
  double sigma_t;    //!< reference macroscopic total at this phase point
  double sigma_a;    //!< reference macroscopic absorption
  double nu_sigma_f; //!< reference macroscopic production
  double sqrtkT;     //!< so the substituted material is evaluated at the
                     //!< host cell's temperature
  double density_mult;
  int32_t cell;
  int32_t material;  //!< reference material index filling that cell
  int64_t seed_id;   //!< identity of the segment; see SourceRoot::seed_id
};

//! Track segments collected during the current generation, one vector per
//! thread. Per-thread rather than shared to avoid an omp critical in the
//! transport loop; merged and sorted at the start of the shadow pass.
extern vector<vector<TrackSite>> thread_tracks;

//! The merged, sorted track segments the shadow pass reads the source from.
extern vector<TrackSite> tracks;

//! A root of one shadow tree: a fission-bank site sampled for the denominator,
//! or one particle of the perturbation source emitted from a track segment.
struct SourceRoot {
  Position r;
  Direction u;
  double E;
  double wgt; //!< expected source weight (always >= 0; the sign is carried by
              //!< which tree it feeds)
  double time;
  int tree;

  //! Seed for the shadow tree grown from this root.
  //!
  //! Derived from the driver's own identity, NOT from this root's position in
  //! the collection: that order depends on thread timing, which would break
  //! reproducibility. The two roots of a +/- PAIR born on the same segment
  //! share this seed (common random numbers), so their descent noise cancels
  //! in the difference -- which is what makes a null perturbation return
  //! exactly zero rather than the difference of two independent estimates.
  int64_t seed_id;
};

//! Roots collected for the current generation, one vector per thread while the
//! source is being emitted, then merged and sorted.
extern vector<vector<SourceRoot>> thread_source_roots;
extern vector<SourceRoot> source_roots;

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

//! Per-batch working sum of `tau` over the current batch's generations, laid
//! out like `tau` ([tree][depth]). Folded into the level accumulators at the
//! batch boundary, then cleared.
extern vector<double> batch_tau;

//! The level estimator's batch statistics, per perturbation and per depth
//! (d = 0..L), accumulated over active batches: sum and sum of squares of the
//! per-batch level l_b(d) = level(d) below. The reported worth and its sigma
//! come from the d = L slice; the whole curve is the depth-convergence
//! diagnostic. This replaces the per-generation tau history -- O(n_pert *
//! (L+1)) scalars, independent of run length.
extern vector<double> ell_sum; //!< [pert * (L+1) + depth]

//! Sum over active batches of l_i(d) * l_j(d), indexed
//! [(i * n_pert + j) * (L+1) + depth]. The diagonal is the sum of squares, so
//! this carries the variance as well; the off-diagonal is what makes the
//! DIFFERENCE between two perturbations far better determined than either one
//! -- they share a driver, a fission source and, where they touch the same
//! cell, their seeds, so most of their noise is common and cancels. Storing
//! only per-perturbation sums would throw that away and force every
//! difference to be quoted with the two variances added.
extern vector<double> ell_cross;

extern int64_t n_active_batches; //!< realizations behind the batch statistics

//! `tau` summed over every active generation of the run, laid out like `tau`
//! ([tree][depth]). The level formed from these pooled totals carries no
//! ratio-of-means bias; it is reported beside the mean of the per-batch levels
//! and is the number to quote if the two ever disagree by a noticeable
//! fraction of sigma.
extern vector<double> pooled_tau;

extern int64_t n_generations;
extern int64_t n_track_total; //!< driver segments seen in a touched cell
extern int64_t n_root_total;  //!< shadow trees grown, all classes

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

//! Record a driver track segment of length `distance` inside `cell_index`.
//!
//! The driver (a BEP_TRUNK neutron) is only read, never altered: this copies
//! its phase point and the reference cross sections it has already computed,
//! and returns. Everything else -- evaluating the substituted material,
//! sampling the source, growing it -- happens in the shadow pass, so nothing
//! but a struct copy sits in the transport hot path and no random number is
//! ever drawn from the driver's streams.
void record_track(Particle& p, int32_t cell_index, double distance);

void score_site(int tree, int super_gen, double wgt);
void run_shadow_pass();
void accumulate_generation();

//! Fold this batch's summed tau into the level's batch statistics. Called at
//! the batch boundary, after the last generation of the batch.
void finalize_batch();

void write_results(hid_t file_id);

//! The depth-d level: the reactivity worth 1/k - 1/k', in absolute units.
//!
//! `d_w`, `fp`, `fn`, `lp`, `ln` are the depth-d descendant weights of the
//! five source-rooted populations and `kk` the reference eigenvalue the
//! fission bank was normalised by (simulation::keff, the same value
//! create_fission_sites() divides by, so the denominator population really is
//! (1/kk) F psi).
//!
//! Writing N_F = fp - fn and N_L = lp - ln, the exact relation between the
//! perturbed-adjoint-weighted sources is
//!
//!     N_F / k'  +  N_L  =  (1/k - 1/k') * <phi'^dag, F psi>  =  drho * kk * D
//!
//! Substituting 1/k' = 1/k - drho and solving for drho gives the closed form
//! below. It is EXACT at any k': no linearisation, no Newton step, no
//! iteration. (A single Newton step, drho ~ R(1-R)/k, agrees only to O(R^2)
//! and is badly wrong for large worths -- -24% at k'/k = 1.49 in the toy.)
//!
//! Both numerator and denominator grow as k'^d with the same coefficient, so
//! the common factor cancels and the level is FLAT in d once the perturbed
//! fundamental mode has established itself. That flatness is the convergence
//! diagnostic, and reading the level at d = L rather than fitting a slope is
//! what removes the fit window and its bias.
inline double level(
  double d_w, double fp, double fn, double lp, double ln, double kk)
{
  double n_f = fp - fn;
  double n_l = lp - ln;
  double den = kk * d_w + n_f;
  if (den == 0.0)
    return 0.0;
  return (n_f / kk + n_l) / den;
}

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
