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

//! Default half-life above which nuclides are treated as stable in [s]
//! (1000 Julian years)
constexpr double DEFAULT_HALF_LIFE_CUTOFF {1000.0 * 365.25 * 24.0 * 3600.0};

//==============================================================================
//! Settings for steady-state decay sources
//==============================================================================

struct DecaySourceSettings {
  //! Lengths of the steps of the irradiation history in [s]. When empty, the
  //! irradiation at constant power is infinitely long.
  vector<double> timesteps;

  //! Relative power during each step. The simulated state corresponds to the
  //! power at the end of the history.
  vector<double> power;

  //! Nuclides with longer half-lives in [s] are treated as stable since they
  //! do not approach equilibrium during operation; their decay is a property
  //! of the material composition rather than of the reaction rates
  double half_life_cutoff {DEFAULT_HALF_LIFE_CUTOFF};

  //! Whether photons, electrons, and positrons are emitted
  array<bool, N_DECAY_PARTICLES> particles {true, true, true};
};

//==============================================================================
//! Irradiation history in the form used by the Bateman solution
//==============================================================================

struct DecayHistory {
  DecayHistory() = default;
  explicit DecayHistory(const DecaySourceSettings& s);

  bool infinite {true}; //!< Whether the irradiation is infinitely long
  double half_life_cutoff {DEFAULT_HALF_LIFE_CUTOFF}; //!< See settings
  vector<double> power;  //!< Power of each step relative to the final power
  vector<double> start;  //!< Time from start of each step to end of history [s]
  vector<double> end;    //!< Time from end of each step to end of history [s]
  double produced {0.0}; //!< Atoms produced per unit final production rate [s]
};

//==============================================================================
//! A linear decay trajectory from a produced radionuclide to an emitting one.
//! The inventory of a radionuclide is the sum over the trajectories that end
//! at it, and the delay between the production of the first nuclide and the
//! decay of the last one is sampled from the trajectory.
//==============================================================================

struct DecayTrajectory {
  int nuclide;               //!< Chain index of the last (emitting) nuclide
  double branching_ratio;    //!< Product of branching ratios along trajectory
  vector<double> lambda;     //!< Decay constants along trajectory [1/s]
  vector<double> step_atoms; //!< Atoms present at the end of the history per
                             //!< unit final production rate from each step [s]
  double atoms;              //!< Sum of step_atoms [s]
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

  //! Kinetic energy emitted per reaction event for each particle type [eV]
  array<double, N_DECAY_PARTICLES> decay_energy {};

  //! Indices in data::decay_trajectories of the trajectories ending at
  //! radionuclides that emit each particle type
  array<vector<int>, N_DECAY_PARTICLES> trajectories;

  //! Distribution for sampling the trajectory
  array<DiscreteIndex, N_DECAY_PARTICLES> trajectory_dist;
};

//==============================================================================
//! Decay source channels of a transport nuclide
//==============================================================================

struct NuclideDecaySource {
  vector<DecaySourceChannel> fission;    //!< Fission channels sorted by energy
  vector<DecaySourceChannel> activation; //!< Activation channels

  //! Whether fission channels produce photons and electrons, which then
  //! replace the delayed photon and beta energy of the fission data
  bool fission_photons {false};
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

//! Irradiation history used by decay sources
extern DecayHistory decay_history;

//! Decay trajectories referenced by decay source channels
extern vector<DecayTrajectory> decay_trajectories;

} // namespace data

//==============================================================================
// Non-member functions
//==============================================================================

//! Read settings for steady-state decay sources and the decay data they need
//! from the depletion chain
//! \param node XML node for <decay_sources>
void read_decay_source_settings(pugi::xml_node node);

//! Find the decay trajectories starting from a produced nuclide
//!
//! The decay network is unfolded with transmutation trajectory analysis (TTA)
//! assuming the nuclide is produced at a rate proportional to the power
//! history and that the radionuclides are removed only by decay. Trajectories
//! ending at stable nuclides are not included.
//!
//! \param i_chain Index of produced nuclide in the depletion chain
//! \param h Irradiation history
//! \return Trajectories with their inventories
vector<DecayTrajectory> find_decay_trajectories(
  int i_chain, const DecayHistory& h);

//! Compute the radionuclide inventory left behind by production of a nuclide
//!
//! \param i_chain Index of produced nuclide in the depletion chain
//! \param s Irradiation history
//! \return Map from chain index to number of atoms present at the end of the
//!   history per unit production rate at the final power, in [s]
std::unordered_map<int, double> decay_inventory(
  int i_chain, const DecaySourceSettings& s);

//! Sample the delay between production and decay along a trajectory
//!
//! The delay follows the distribution of the ages of the atoms that decay at
//! the end of the history, i.e., the Bateman kernel of the trajectory
//! weighted by the power history.
//!
//! \param traj Decay trajectory
//! \param h Irradiation history the trajectory was computed with
//! \param seed Pseudorandom number seed pointer
//! \return Delay in [s]
double sample_decay_delay(
  const DecayTrajectory& traj, const DecayHistory& h, uint64_t* seed);

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

//! Change in the neutron heating number due to decay sources
//!
//! With photon transport, decay electrons replace the delayed beta energy of
//! the fission heating number. Without photon transport, no particles are
//! emitted and the decay energy is deposited locally: for 'heating', the
//! kinetic energy of decay electrons and positrons replaces the delayed beta
//! energy, and for 'heating-local', the energy of all decay particles,
//! including positron annihilation, replaces the delayed beta and photon
//! energy. Activation channels add the decay energy of activation products.
struct DecayHeating {
  double fission {0.0};    //!< Change in fission heating [eV-b]
  double activation {0.0}; //!< Heating from activation products [eV-b]
};

//! \param p Particle that is colliding
//! \param i_nuclide Index of nuclide
//! \param mt MT of the heating score (HEATING or HEATING_LOCAL)
//! \return Change in heating number
DecayHeating decay_source_heating(const Particle& p, int i_nuclide, int mt);

void free_memory_decay_sources();

} // namespace openmc

#endif // OPENMC_DECAY_SOURCE_H
