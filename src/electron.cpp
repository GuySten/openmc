#include "openmc/electron.h"

#include "openmc/array.h"
#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"
#include "openmc/distribution_multi.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/photon.h"
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

std::unordered_map<std::string, int> electron_map;
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
  data::electron_map[name_] = index_;

  // Resolve the index of this element in data::photoatomic, which is needed
  // for subshell binding energies and atomic relaxation. The photoatomic data
  // for an element is always loaded immediately before its electron data, so
  // the entry exists by now. Do not assume the two indices coincide.
  auto it = data::element_map.find(name_);
  if (it == data::element_map.end()) {
    fatal_error(fmt::format("Photoatomic data for element {} must be loaded "
                            "before its electron data.",
      name_));
  }
  i_photoatomic_ = it->second;

  // Get atomic number
  read_attribute(group, "Z", Z_);

  // Determine number of energies and read energy grid
  read_dataset(group, "energy", energy_);

  // Read elastic scattering
  hid_t rgroup = open_group(group, "elastic");
  read_dataset(rgroup, "xs", elastic_);
  hid_t dist_group = open_group(rgroup, "distribution");
  elastic_angle_ = AngleDistribution {dist_group};
  close_group(dist_group);
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
  vector<std::string> designators;
  read_attribute(rgroup, "designators", designators);
  for (auto designator : designators) {
    hid_t shell_group = open_group(rgroup, designator.c_str());
    hid_t egroup = open_group(shell_group, "energy");
    ionization_dist_.push_back(make_unique<ContinuousTabular>(egroup));
    close_group(egroup);
    close_group(shell_group);
  }
  close_group(rgroup);

  // Map each electroionization subshell onto the corresponding subshell of the
  // photoatomic data, which holds the binding energies and relaxation
  // transitions. Matching is by ENDF designator: the electroionization list
  // (NXS(7) subshells) and the photoatomic list need not agree in length or
  // order, and in particular neither corresponds to the Compton Doppler
  // broadening shell list (NXS(5) shells).
  const auto& photoatomic {*data::photoatomic[i_photoatomic_]};
  shell_map_.resize(designators.size(), -1);
  for (int i = 0; i < designators.size(); ++i) {
    int endf_index = 0;
    int j = 1;
    for (const auto& subshell : SUBSHELLS) {
      if (designators[i] == subshell) {
        endf_index = j;
        break;
      }
      ++j;
    }

    for (int k = 0; k < photoatomic.shells_.size(); ++k) {
      if (photoatomic.shells_[k].index_subshell == endf_index) {
        shell_map_[i] = k;
        break;
      }
    }

    if (shell_map_[i] < 0) {
      fatal_error(fmt::format(
        "Electroionization subshell {} of element {} has no counterpart in the "
        "photoatomic data, so its binding energy and relaxation transitions "
        "are unavailable. The electron and photon libraries are inconsistent.",
        designators[i], name_));
    }
  }

  // Read bremsstrahlung
  rgroup = open_group(group, "bremsstrahlung");
  read_dataset(rgroup, "xs", bremsstrahlung_);
  dist_group = open_group(rgroup, "distribution");
  hid_t egroup = open_group(dist_group, "energy");
  bremsstrahlung_dist_ = make_unique<ContinuousTabular>(egroup);
  close_group(egroup);
  close_group(dist_group);
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
  const auto ion_i = ionization_.slice(tensor::all, i_grid).sum();
  const auto ion_ip1 = ionization_.slice(tensor::all, i_grid + 1).sum();
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
  return elastic_angle_.sample(E, seed);
}

double ElectronInteraction::excitation(double E) const
{
  return E - excitation_energy_loss_(E);
}

void ElectronInteraction::ionization(Particle& p, int i_shell) const
{
  double E_knock = ionization_dist_[i_shell]->sample(p.E(), p.current_seed());
  double phi = uniform_distribution(0., 2.0 * PI, p.current_seed());
  // Binding energies live on the photoatomic subshell list. Note this is NOT
  // PhotonInteraction::binding_energy_, which belongs to the shorter Compton
  // Doppler broadening shell list and would be indexed out of bounds here.
  const auto& element {*data::photoatomic[i_photoatomic_]};
  double e_b = element.shells_[shell_map_[i_shell]].binding_energy;

  // The scattered primary must be left with positive energy. A sampled
  // knock-on energy that violates this would give a negative energy electron
  // and a negative argument in the scattering cosine below.
  if (E_knock + e_b >= p.E()) {
    p.write_restart();
    fatal_error(fmt::format(
      "Electroionization of {} shell {} at {} eV sampled a knock-on energy of "
      "{} eV which, with a binding energy of {} eV, exceeds the energy of the "
      "incident electron.",
      name_, i_shell, p.E(), E_knock, e_b));
  }

  double mu_knock = std::sqrt((1.0 + 2.0 * MASS_ELECTRON_EV / p.E()) /
                              (1.0 + 2.0 * MASS_ELECTRON_EV / E_knock));
  Direction u_knock = rotate_angle(p.u(), mu_knock, &phi, p.current_seed());
  p.create_secondary(p.wgt(), u_knock, E_knock, ParticleType::electron());

  p.mu() = std::sqrt((1.0 + 2.0 * MASS_ELECTRON_EV / p.E()) /
                     (1.0 + 2.0 * MASS_ELECTRON_EV / (p.E() - E_knock - e_b)));
  phi += PI;
  p.u() = rotate_angle(p.u(), p.mu(), &phi, p.current_seed());
  p.E() = p.E() - E_knock - e_b;
}

int ElectronInteraction::sample_ionization_shell(Particle& p) const
{
  auto& xs {p.electron_xs(index_)};

  // Sample cumulative distribution function
  double cutoff = prn(p.current_seed()) * xs.ionization;
  int n_shell = ionization_.shape(0);
  int i_grid = xs.index_grid;
  double f = xs.interp_factor;

  int i_shell;
  double prob = 0.0;
  for (i_shell = 0; i_shell < n_shell; ++i_shell) {
    double sigma =
      ionization_(i_shell, i_grid) +
      f * (ionization_(i_shell, i_grid + 1) - ionization_(i_shell, i_grid));
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
  double E_photon = bremsstrahlung_dist_->sample(p.E(), p.current_seed());
  p.E() -= E_photon;
  p.create_secondary(p.wgt(), p.u(), E_photon, ParticleType::photon());
}

//==============================================================================
// Non-member functions
//==============================================================================

void free_memory_electron()
{
  data::electroatomic.clear();
  data::electron_map.clear();
}

} // namespace openmc
