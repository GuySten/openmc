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
//!  - probe roots (optional): at every recorded fission event the shadow pass
//!    emits one probe photon, at one of a comb of line energies picked
//!    uniformly, with unit intensity per line per fission. It is a shadow
//!    particle, labelled 0 while uncollided, 1 after coherent scattering
//!    only, and 2 after an energy-changing collision (and on every secondary
//!    photon it makes). Its photoneutrons become probe roots, tagged by
//!    fissioning nuclide, line and label, so that the importance of a photon
//!    of any energy can be rebuilt from the lines.
//!  - ray probes (optional, with probes): at every recorded fission event
//!    (rouletted) an uncollided ray is walked through the geometry; every
//!    line's uncollided photoneutron production along it is exact, and one
//!    tree, started at a comb energy of photoneutron birth energies, is
//!    shared by every line with its weight. Label 0 is then the rays'.
//!
//! With perturbed importance on, fission- and delayed-root trees also
//! transport photons below their root, and every photoneutron those photons
//! make grows on as a branch of its tree, scored in a class of its own. The
//! tree's weight without its branches is the unperturbed importance, and with
//! them the importance with photoneutrons in the chain, to first order.
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

//! Root populations, and the perturbation branches scored beside them
enum RootClass : int {
  CLASS_FISSION = 0,        //!< sample of the fission bank (the denominator)
  CLASS_DELAYED = 1,        //!< forced expected-value delayed neutrons
  CLASS_PHOTONEUTRON = 2,   //!< photoneutrons diverted from the driver
  CLASS_FISSION_BRANCH = 3, //!< photoneutron branches of fission-root trees
  CLASS_DELAYED_BRANCH = 4, //!< photoneutron branches of delayed-root trees
  CLASS_PROBE = 5,          //!< photoneutrons of probe photons
  CLASS_RAY = 6             //!< shared photoneutrons of ray probes
};
constexpr int N_CLASS = 3;       //!< root populations with class targets
constexpr int N_SCORE_CLASS = 5; //!< classes in the weight arrays
//! Classes a shadow tag can carry. Probe trees score in arrays of their own,
//! so the weight arrays keep N_SCORE_CLASS classes.
constexpr int N_PACK_CLASS = 7;

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
  int ebin {-1}; //!< photoneutron energy group, -1 if none
  int64_t seed_id;
};

//! A shadow tag packs the class, the delayed-group tag and, for photoneutron
//! roots with energy groups on, the energy group + 1 (0: none), so that the
//! group rides down the whole tree. With groups off the tag is unchanged.
inline int tag_class(int shadow_tag)
{
  return (shadow_tag / N_TAG) % N_PACK_CLASS;
}
inline int tag_ebin(int shadow_tag)
{
  return shadow_tag / (N_TAG * N_PACK_CLASS) - 1;
}
inline int pack_tag(int cls, int tag, int ebin)
{
  return cls * N_TAG + tag + (ebin + 1) * N_TAG * N_PACK_CLASS;
}

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

//! Probe-photon hook for a photoneutron leaving a probe photon (or one of its
//! secondary photons): record it as a probe root.
//! \return true if the photoneutron was taken as a root and must not be
//! banked as a secondary
bool record_probe_photoneutron(Particle& p, double wgt, Direction u, double E);

//! Probe-photon hook after an energy-changing scattering: label 2
void mark_scattered(Particle& p);

//! Probe-photon hook after a coherent scattering: label 1 if uncollided
void mark_coherent(Particle& p);

//! Shadow tag of a secondary photon made by a particle with this tag: a
//! probe's secondary photons carry label 2
int secondary_photon_tag(int shadow_tag);

//! Shadow-tree hook in create_fission_sites(): bank this tree's fission
//! sites at its population's target weight and score them by depth
void create_tree_sites(Particle& p, int i_nuclide, const Reaction& rx);

//! Does this shadow-tree neutron emit photons? Only with perturbed importance
//! on, and only below the root (depth >= 1) of fission- and delayed-root
//! trees: photoneutrons of the root itself are the driver's photoneutron
//! population, and branches of branches are second order.
bool tree_makes_photons(const Particle& p);

//! Shadow-tree hook for a photoneutron leaving a tree photon: roulette it to
//! its branch's target weight.
//! \param[inout] wgt photoneutron weight, set to the weight it is banked with
//! \param[out] tag shadow tag the banked photoneutron is to carry
//! \return false if the photoneutron was killed by the roulette
bool branch_photoneutron(Particle& p, double& wgt, int& tag);

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
