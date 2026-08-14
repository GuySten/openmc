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
  hid_t dset = open_dataset(rgroup, "energy_loss");
  excitation_energy_loss_ = Tabulated1D {dset};
  close_dataset(dset);
  close_group(rgroup);

  // Read ionization
  rgroup = open_group(group, "ionization");
  read_dataset(rgroup, "xs", ionization_);
  close_group(rgroup);

  // Read bremsstrahlung
  rgroup = open_group(group, "bremsstrahlung");
  read_dataset(rgroup, "xs", bremsstrahlung_);
  close_group(rgroup);
}

void ElectronInteraction::calculate_xs(Particle& p) const
{
  // Perform binary search on the element energy grid in order to determine
  // which points to interpolate between
  int n_grid = energy_.size();
  double E = p.E();
  int i_grid;
  if (E <= energy_[0]) {
    i_grid = 0;
  } else if (E > energy_(n_grid - 1)) {
    i_grid = n_grid - 2;
  } else {
    // We use upper_bound_index here because sometimes photons are created with
    // energies that exactly match a grid point
    i_grid = upper_bound_index(energy_.cbegin(), energy_.cend(), E);
  }

  // check for case where two energy points are the same
  if (energy_(i_grid) == energy_(i_grid + 1))
    ++i_grid;

  // calculate interpolation factor
  double f = (E - energy_(i_grid)) / (energy_(i_grid + 1) - energy_(i_grid));

  auto& xs {p.electron_xs(index_)};
  xs.index_grid = i_grid;
  xs.interp_factor = f;

  // Calculate microscopic elastic cross section
  xs.elastic = elastic_(i_grid) + f * (elastic_(i_grid + 1) - elastic_(i_grid));

  // Calculate microscopic excitation cross section
  xs.excitation =
    excitation_(i_grid) + f * (excitation_(i_grid + 1) - excitation_(i_grid));

  // Calculate microscopic ionization cross section
  const auto ion_i = ionization_.slice(i_grid, tensor::all).sum();
  const auto ion_ip1 = ionization_.slice(i_grid + 1, tensor::all).sum();
  xs.ionization = ion_i + f * (ion_ip1 - ion_i);

  // Calculate microscopic bremsstrahlung cross section
  xs.bremsstrahlung =
    bremsstrahlung_(i_grid) +
    f * (bremsstrahlung_(i_grid + 1) - bremsstrahlung_(i_grid));

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
  return E - excitation_energy_loss_(E);
}

void ElectronInteraction::ionization(Particle& p, int i_shell) const
{
  return;
}

int ElectronInteraction::sample_ionization_shell(Particle& p) const
{
  auto& xs {p.electron_xs(index_)};

  // Sample cumulative distribution function
  double cutoff = prn(p.current_seed()) * xs.ionization;
  int n_shell = ionization_.shape(1);
  int i_grid = xs.index_grid;
  double f = xs.interp_factor;

  int i_shell;
  double prob = 0.0;
  for (i_shell = 0; i_shell < n_shell; ++i_shell) {
    double sigma =
      ionization_(i_grid, i_shell) +
      f * (ionization_(i_grid + 1, i_shell) - ionization_(i_grid, i_shell));
    // Increment probability to compare to cutoff
    prob += sigma;
    if (prob > cutoff)
      return i_shell;
  }

  // If we made it here, no shell was sampled
  p.write_restart();
  fatal_error("Did not sample any electron shell during electro-ionization.");
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
