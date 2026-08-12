#include "openmc/electron.h"

#include "openmc/array.h"
#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"
#include "openmc/distribution_multi.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/physics.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/settings.h"

#include "openmc/tensor.h"

#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <stdexcept>
#include <tuple> // for tie

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace data {

vector<unique_ptr<ElectronInteraction>> electroatomic;

} // namespace data

//==============================================================================
// ElectronInteraction implementation
//==============================================================================

ElectronInteraction::ElectronInteraction(hid_t group)
{
  // Set index of element in global vector
  index_ = data::electroatomic.size();

  // Get name of nuclide from group, removing leading '/'
  name_ = object_name(group).substr(1);
  data::element_map[name_] = index_;

  // Get atomic number
  read_attribute(group, "Z", Z_);

  // Determine number of energies and read energy grid
  read_dataset(group, "energy", energy_);

  // Read elastic scattering
  hid_t rgroup = open_group(group, "elastic");
  read_dataset(rgroup, "xs", elastic_);
  close_group(rgroup);

  // Read excitation
  rgroup = open_group(group, "excitation");
  read_dataset(rgroup, "xs", excitation_);
  read_dataset(rgroup, "energy_loss", excitation_energy_loss_);
  close_group(rgroup);

  // Read ionization
  rgroup = open_group(group, "ionization");
  read_dataset(rgroup, "xs", ionization_);
  close_group(rgroup);

  // Read bremsstrahlung
  rgroup = open_group(group, "bremsstrahlung");
  read_dataset(rgroup, "xs", bremsstrahlung_);
  close_group(rgroup);

  // Take logarithm of energies and cross sections since they are log-log
  // interpolated. Note that cross section libraries converted from ACE files
  // represent zero as exp(-500) to avoid log-log interpolation errors. For
  // values below exp(-499) we store the log as -900, for which exp(-900)
  // evaluates to zero.
  double limit = std::exp(-499.0);
  energy_ = tensor::log(energy_);
  elastic_ = tensor::where(elastic_ > limit, tensor::log(elastic_), -900.0);
  excitation_ =
    tensor::where(excitation_ > limit, tensor::log(excitation_), -900.0);
  excitation_energy_loss_ = tensor::where(excitation_energy_loss_ > limit,
    tensor::log(excitation_energy_loss_), -900.0);
  ionization_ =
    tensor::where(ionization_ > limit, tensor::log(ionization_), -900.0);
  bremsstrahlung_ = tensor::where(
    bremsstrahlung_ > limit, tensor::log(bremsstrahlung_), -900.0);
}

void ElectronInteraction::calculate_xs(Particle& p) const
{
  // Perform binary search on the element energy grid in order to determine
  // which points to interpolate between
  int n_grid = energy_.size();
  double log_E = std::log(p.E());
  int i_grid;
  if (log_E <= energy_[0]) {
    i_grid = 0;
  } else if (log_E > energy_(n_grid - 1)) {
    i_grid = n_grid - 2;
  } else {
    // We use upper_bound_index here because sometimes photons are created with
    // energies that exactly match a grid point
    i_grid = upper_bound_index(energy_.cbegin(), energy_.cend(), log_E);
  }

  // check for case where two energy points are the same
  if (energy_(i_grid) == energy_(i_grid + 1))
    ++i_grid;

  // calculate interpolation factor
  double f =
    (log_E - energy_(i_grid)) / (energy_(i_grid + 1) - energy_(i_grid));

  auto& xs {p.electron_xs(index_)};
  xs.index_grid = i_grid;
  xs.interp_factor = f;

  // Calculate microscopic elastic cross section
  xs.elastic =
    std::exp(elastic_(i_grid) + f * (elastic_(i_grid + 1) - elastic_(i_grid)));

  // Calculate microscopic excitation cross section
  xs.excitation = std::exp(
    excitation_(i_grid) + f * (excitation_(i_grid + 1) - excitation_(i_grid)));

  // Calculate microscopic ionization cross section
  xs.ionization = std::exp(
    ionization_(i_grid) + f * (ionization_(i_grid + 1) - ionization_(i_grid)));

  // Calculate microscopic bremsstrahlung cross section
  xs.bremsstrahlung =
    std::exp(bremsstrahlung_(i_grid) +
             f * (bremsstrahlung_(i_grid + 1) - bremsstrahlung_(i_grid)));

  // Calculate microscopic total cross section
  xs.total = xs.elastic + xs.excitation + xs.ionization + xs.bremsstrahlung;
  xs.last_E = p.E();
}

double ElectronInteraction::elastic_scatter(double E, uint64_t* seed) const
{
  double mu;
  return mu;
}

double ElectronInteraction::excitation(double E) const
{
  double E_out;
  return E_out;
}

void ElectronInteraction::ionization(Particle& p, int i_shell) const
{
  return;
}

int ElectronInteraction::sample_ionization_shell(double E, uint64_t* seed) const
{
  int i_shell;
  return i_shell;
}

void ElectronInteraction::bremsstrahlung(Particle& p) const
{
  return;
}

//==============================================================================
// Non-member functions
//==============================================================================

void free_memory_electron()
{
  data::electroatomic.clear();
}

} // namespace openmc
