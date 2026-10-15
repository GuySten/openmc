//! \file decay_source.h
//! \brief Steady-state sources of decay photons, electrons, and positrons

#ifndef OPENMC_DECAY_SOURCE_H
#define OPENMC_DECAY_SOURCE_H

#include <string>
#include <unordered_map>

#include "pugixml.hpp"

#include "openmc/array.h"
#include "openmc/distribution.h"
#include "openmc/particle.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Constants
//==============================================================================

//! Number of particle types that can be emitted (photon, electron, positron)
constexpr int N_DECAY_PARTICLES {3};

//! Default irradiation time in [s] (one Julian year)
constexpr double DEFAULT_IRRADIATION_TIME {365.25 * 24.0 * 3600.0};

//==============================================================================
//! Settings for steady-state decay sources
//==============================================================================

struct DecaySourceSettings {
  //! Lengths of the steps of the irradiation history in [s]
  vector<double> timesteps {DEFAULT_IRRADIATION_TIME};

  //! Relative power during each step. The simulated state corresponds to the
  //! power at the end of the history.
  vector<double> power {1.0};

  //! Whether photons, electrons, and positrons are emitted
  array<bool, N_DECAY_PARTICLES> particles {true, true, true};
};

//==============================================================================
//! A reaction that produces radionuclides whose decay emits particles. At
//! steady state, each reaction event is accompanied by the decays of the
//! radionuclides it leaves behind.
//==============================================================================

struct DecaySourceChannel {
  bool fission {false};  //!< Whether this is a fission channel
  double energy {0.0};   //!< Incident energy of fission yields in [eV]
  vector<int> reactions; //!< Indices of reactions giving the cross section
  int mt {0};            //!< MT of the reaction

  //! Particles emitted per reaction event for each particle type
  array<double, N_DECAY_PARTICLES> yield {};

  //! Chain indices of emitting radionuclides for each particle type
  array<vector<int>, N_DECAY_PARTICLES> nuclides;

  //! Distribution for sampling the emitting radionuclide
  array<DiscreteIndex, N_DECAY_PARTICLES> nuclide_dist;
};

//==============================================================================
//! Decay source channels of a transport nuclide
//==============================================================================

struct NuclideDecaySource {
  vector<DecaySourceChannel> fission;    //!< Fission channels sorted by energy
  vector<DecaySourceChannel> activation; //!< Activation channels

  //! Whether fission product decay electrons are emitted explicitly, in which
  //! case delayed beta energy is removed from the fission heating
  bool fission_electrons {false};

  bool empty() const { return fission.empty() && activation.empty(); }
};

//==============================================================================
// Global variables
//==============================================================================

namespace settings {

extern bool decay_sources_on; //!< Whether steady-state decay sources are used
extern DecaySourceSettings decay_source_settings;

} // namespace settings

namespace data {

//! Decay sources for each transport nuclide
extern vector<NuclideDecaySource> decay_sources;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

//! Read settings for steady-state decay sources
//! \param node XML node for <decay_sources>
void read_decay_source_settings(pugi::xml_node node);

//! Compute the radionuclide inventory left behind by production of a nuclide
//!
//! The inventory is computed with transmutation trajectory analysis (TTA) of
//! the decay chain, assuming the nuclide is produced at a rate proportional
//! to the power history and that the radionuclides are removed only by decay.
//!
//! \param i_chain Index of produced nuclide in the depletion chain
//! \param s Irradiation history
//! \return Map from chain index to number of atoms present at the end of the
//!   history per unit production rate at the final power, in [s]
std::unordered_map<int, double> decay_inventory(
  int i_chain, const DecaySourceSettings& s);

//! Whether decay sources produce delayed fission photons for a nuclide, in
//! which case they replace scaling of the prompt fission photon yield
//! \param name Name of the fissionable nuclide
//! \return Whether fission product yields are available in the chain
bool decay_sources_produce_fission_photons(const std::string& name);

//! Build decay source channels for all loaded nuclides
void finalize_decay_sources();

//! Create decay photons, electrons, and positrons at a neutron collision
//! \param p Particle that is colliding
//! \param i_nuclide Index of nuclide that the particle collided with
void sample_decay_sources(Particle& p, int i_nuclide);

//! Delayed beta heating to remove from the fission heating number
//! \param p Particle that is colliding
//! \param i_nuclide Index of nuclide
//! \return Delayed beta heating in [eV-b]
double decay_source_beta_heating(const Particle& p, int i_nuclide);

void free_memory_decay_sources();

} // namespace openmc

#endif // OPENMC_DECAY_SOURCE_H
