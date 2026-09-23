//! \file adjoint_populations.h
//! \brief Adjoint side populations: importance-weighted kinetics parameters
//! and photoneutron effects from shadow trees grown beside the driver.
//!
//! The driver is an ordinary eigenvalue calculation and is not altered. It
//! only records roots:
//!
//!  - fission roots: a sample of each generation's fission bank, tagged with
//!    the delayed group of the site (0 = prompt);
//!  - delayed roots: at every fission-site creation, the expected number of
//!    delayed neutrons of each group, w/k * (sigma_f/sigma_t) * nu_d,g(E), so
//!    that the delayed fraction is estimated without waiting for rare
//!    delayed sites;
//!  - photoneutron roots (optional): photoneutrons from driver photons,
//!    diverted here instead of being banked as secondaries, so they never
//!    enter the fission chain. Tagged with the photofission delayed group
//!    (0 = prompt).
//!
//! After each active generation every root is Russian-rouletted to its
//! population's target weight and grown n_generation generations in the
//! unperturbed physics, as a neutron-only shadow tree that banks its fission
//! sites at the same target weight. Its weight at depth L is the root's
//! importance (iterated fission probability). A depth-1 site also carries
//! the lifetime its root had reached at the fission that produced it, so
//! that sum(w * t0) / (k * sum(w)) over the fission roots is IFP's
//! generation time.
//!
//! Per batch, and per population, tag and depth, the module accumulates the
//! summed weight and the summed weight times lifetime stamp. Ratios --
//! beta_eff per group, generation time, the photoneutron fraction and its
//! effect on both -- are formed from these sums in the Python API, with
//! ordinary batch statistics.

#ifndef OPENMC_ADJOINT_POPULATIONS_H
#define OPENMC_ADJOINT_POPULATIONS_H

#include <cstdint>

#include "hdf5.h"

#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {

class Particle;
class Reaction;

namespace adjpop {

//! Root populations
enum RootClass : int {
  CLASS_FISSION = 0,     //!< sample of the fission bank (the denominator)
  CLASS_DELAYED = 1,     //!< forced expected-value delayed neutrons
  CLASS_PHOTONEUTRON = 2 //!< photoneutrons diverted from the driver
};
constexpr int N_CLASS = 3;

//! Tags are 0 (prompt) or a delayed group 1..N_TAG-1
constexpr int N_TAG = 9;

//! Fraction of the particles per rank each population is rouletted down to per
//! generation. Every population then costs about this fraction of a driver
//! generation per tree generation.
constexpr double ROOT_FRACTION = 0.1;

//! A root waiting to be grown
struct Root {
  Position r;
  Direction u;
  double E;
  double time;
  double wgt; //!< raw weight, before the roulette
  int cls;
  int tag;
  int64_t seed_id;
};

//! A driver fission-site creation, from which delayed roots are emitted in
//! the shadow pass (so that the driver draws no extra random numbers)
struct FissionRecord {
  Position r;
  double E;
  double time;
  double wgt;          //!< p.wgt() / keff
  double sigma_f_to_t; //!< microscopic sigma_f / sigma_t of the nuclide
  const Reaction* rx;
  int i_nuclide;
  int64_t seed_id;
};

//==============================================================================
// Hooks
//==============================================================================

//! Allocate storage; called once after settings are read
void init();

//! Decide whether this generation grows side populations and clear the
//! per-generation records. Called at the start of each generation.
void reset_generation();

//! Driver hook in create_fission_sites(): record a fission event for the
//! delayed roots. Draws no random numbers.
void record_fission(Particle& p, int i_nuclide, const Reaction& rx);

//! Driver hook for a photoneutron leaving a photon: record it as a root.
//! \return true if the photoneutron was taken as a root and must not be
//! banked as a secondary
bool record_photoneutron(
  Particle& p, double wgt, Direction u, double E, int delayed_group);

//! Shadow-tree hook in create_fission_sites(): bank this tree's fission
//! sites at its population's target weight and score them by depth
void create_tree_sites(Particle& p, int i_nuclide, const Reaction& rx);

//! Grow every root recorded this generation. Called from
//! finalize_generation() once the fission bank is complete.
void run_shadow_pass();

//! Fold this batch's sums into the per-batch record
void finalize_batch();

//! Write the per-batch sums to a statepoint
void write_results(hid_t file_id);

//! Free storage
void clear();

} // namespace adjpop

namespace simulation {
extern bool adjpop_on; //!< growing side populations this generation?
} // namespace simulation

} // namespace openmc

#endif // OPENMC_ADJOINT_POPULATIONS_H
