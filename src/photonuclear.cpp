#include "openmc/photonuclear.h"

#include "openmc/array.h"
#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"
#include "openmc/distribution_multi.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/settings.h"

#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xoperation.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xview.hpp"

#include <cmath>
#include <fmt/core.h>
#include <tuple> // for tie

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace data {

//! Photonuclear interaction data for each isotope
std::unordered_map<std::string, int> photonuclear_map;
vector<unique_ptr<PhotonuclearInteraction>> photonuclears;
double photonuclear_energy_min;

} // namespace data

//==============================================================================
// PhotonuclearInteraction implementation
//==============================================================================

PhotonuclearInteraction::PhotonuclearInteraction(hid_t group)
{
  using namespace xt::placeholders;

  // Set index of element in global vector
  index_ = data::photonuclears.size();
  
  // Get name of nuclide from group, removing leading '/'
  name_ = object_name(group).substr(1);
  data::photonuclear_map[name_] = index_;

  read_attribute(group, "Z", Z_);
  read_attribute(group, "A", A_);
  read_attribute(group, "metastable", metastable_);
  read_attribute(group, "atomic_weight_ratio", awr_);
  
  

  // Determine number of energies and read energy grid
  read_dataset(group, "energy", energy_);
  
  hid_t rxs_group = open_group(group, "reactions");
  
  // Read disappearance
  hid_t rgroup = open_group(rxs_group, "reaction_003");
  read_dataset(rgroup, "xs", disappearance_);
  

  // Read heating
  if (object_exists(rxs_group, "reaction_301")) {
    rgroup = open_group(rxs_group, "reaction_301");
    read_dataset(rgroup, "xs", heating_);
    close_group(rgroup);
  } else {
    heating_ = xt::zeros_like(energy_);
  }
  close_group(rgroup);
  close_group(rxs_group);
}

PhotonuclearInteraction::~PhotonuclearInteraction()
{
  data::photonuclear_map.erase(name_);
}

void PhotonuclearInteraction::calculate_xs(Particle& p) const
{
  // Perform binary search on the element energy grid in order to determine
  // which points to interpolate between
  int n_grid = energy_.size();
  double E = p.E();
  int i_grid;
  if (E < energy_[0]) {
    auto& xs {p.photonuclear_xs(index_)};
    xs.index_grid = -1;
    xs.heating = 0.0;
    xs.disappearance = 0.0;
    xs.last_E = p.E();
    return;
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

  auto& xs {p.photonuclear_xs(index_)};
  xs.index_grid = i_grid;
  xs.interp_factor = f;

  // Calculate microscopic coherent cross section
  xs.disappearance = disappearance_(i_grid) + f * (disappearance_(i_grid + 1) - disappearance_(i_grid));

  // Calculate microscopic heating cross section
  xs.heating = heating_(i_grid) + f * (heating_(i_grid + 1) - heating_(i_grid));

  xs.last_E = p.E();
}

//==============================================================================
// Non-member functions
//==============================================================================


void free_memory_photonuclear()
{
  data::photonuclears.clear();
}

} // namespace openmc
