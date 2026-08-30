#ifndef OPENMC_PHYSICS_H
#define OPENMC_PHYSICS_H

#include "openmc/bank.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/photonuclear.h"
#include "openmc/position.h"
#include "openmc/reaction.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Non-member functions
//==============================================================================

//! Sample a nuclide and reaction and then calls the appropriate routine
void collision(Particle& p);

//! Samples an incident neutron reaction
void sample_neutron_reaction(Particle& p);

//! Samples an element based on the macroscopic cross sections for each nuclide
//! within a material and then samples a reaction for that element and calls the
//! appropriate routine to process the physics.
void sample_photon_reaction(Particle& p);

//! Terminates the particle and either deposits all energy locally
//! (electron_treatment = ElectronTreatment::LED) or creates secondary
//! bremsstrahlung photons from electron deflections with charged particles
//! (electron_treatment = ElectronTreatment::TTB).
void sample_electron_reaction(Particle& p);

//! Terminates the particle and either deposits all energy locally
//! (electron_treatment = ElectronTreatment::LED) or creates secondary
//! bremsstrahlung photons from electron deflections with charged particles
//! (electron_treatment = ElectronTreatment::TTB). Two annihilation photons of
//! energy MASS_ELECTRON_EV (0.511 MeV) are created and travel in opposite
//! directions.
void sample_positron_reaction(Particle& p);

//! Process a secondary charged particle at its birth site
void process_charged_secondary(
  Particle& p, Direction u, double E, ParticleType type);

//! Sample a nuclide based on their total cross sections and densities within
//! the current material
//!
//! \param[in] p Particle
//! \return Index in the data::nuclides vector
int sample_nuclide(Particle& p);

//! Sample a nuclide for photonuclear interaction based on their
//! cross sections and densities within the current material
//!
//! \param[in] p Particle
//! \param[in] biased If true, sample proportional to the neutron production
//!   cross section (for the forced photoneutron); otherwise proportional to
//!   the total photonuclear cross section (for the analog absorption)
//! \return Index in the data::photonuclears vector
int sample_photonuclear_nuclide(Particle& p, bool biased);

void sample_photoneutron_product(
  int i_nuclide, Particle& p, int* i_rx, int* i_product);

//! Determine the average total, prompt, and delayed neutrons produced from
//! fission and creates appropriate bank sites.
void create_fission_sites(Particle& p, int i_nuclide, const Reaction& rx);

int sample_photon_element(Particle& p);

int sample_electron_element(Particle& p);

Reaction& sample_fission(int i_nuclide, Particle& p);

void sample_photon_product(
  int i_nuclide, Particle& p, int* i_rx, int* i_product);

void absorption(Particle& p, int i_nuclide);

void scatter(Particle& p, int i_nuclide);

//! Treats the elastic scattering of a neutron with a target.
void elastic_scatter(int i_nuclide, const Reaction& rx, double kT, Particle& p);

void sab_scatter(int i_nuclide, int i_sab, Particle& p);

//! samples the target velocity. The constant cross section free gas model is
//! the default method. Methods for correctly accounting for the energy
//! dependence of cross sections in treating resonance elastic scattering such
//! as the DBRC and a new, accelerated scheme are also implemented here.
Direction sample_target_velocity(const Nuclide& nuc, double E, Direction u,
  Direction v_neut, double xs_eff, double kT, uint64_t* seed);

//! samples a target velocity based on the free gas scattering formulation, used
//! by most Monte Carlo codes, in which cross section is assumed to be constant
//! in energy. Excellent documentation for this method can be found in
//! FRA-TM-123.
Direction sample_cxs_target_velocity(
  double awr, double E, Direction u, double kT, uint64_t* seed);

void sample_fission_neutron(
  int i_nuclide, const Reaction& rx, SourceSite* site, Particle& p);

//! handles all reactions with a single secondary neutron (other than fission),
//! i.e. level scattering, (n,np), (n,na), etc.
void inelastic_scatter(const Nuclide& nuc, const Reaction& rx, Particle& p);

void sample_secondary_photons(Particle& p, int i_nuclide);

//! Handle an analog photonuclear absorption. The photon is absorbed and its
//! secondary products are banked. Photoneutrons are emitted here only when
//! settings::photoneutron_biasing is off.
void photonuclear_collision(Particle& p);

//! Emit a single photoneutron carrying the expected weight. Called at every
//! photon collision when settings::photoneutron_biasing is on.
void emit_forced_photoneutron(Particle& p);

//! Emit a single photofission neutron, choosing prompt or delayed emission and
//! advancing the emission time for delayed precursors.
double emit_photofission_neutron(Particle& p,
  const PhotonuclearInteraction& nuc, const PhotonuclearReaction& rx,
  double wgt);

//! Sample the energy and direction of a single photonuclear reaction product
//! and bank it. Returns the sampled outgoing energy in [eV].
double emit_photonuclear_product(Particle& p,
  const PhotonuclearInteraction& nuc, const PhotonuclearReaction& rx,
  const ReactionProduct& product, double wgt);

//! Split or Roulette particles based their weight and the lower weight window
//! bound.
//
//! \param[in] p, particle to be split or rouletted with the weight window.
void split_particle(Particle& p);

} // namespace openmc

#endif // OPENMC_PHYSICS_H
