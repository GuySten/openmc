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
#include "openmc/string_utils.h"

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
// PhotonuclearReaction implementation
//==============================================================================

PhotonuclearReaction::PhotonuclearReaction(
  hid_t group, std::string name)
{
  read_attribute(group, "Q_value", q_value_);
  read_attribute(group, "mt", mt_);
  int tmp;
  read_attribute(group, "center_of_mass", tmp);
  scatter_in_cm_ = (tmp == 1);

  // Checks if redudant attribute exists before loading
  // (for compatibiltiy with legacy .h5 libraries)
  if (attribute_exists(group, "redundant")) {
    read_attribute(group, "redundant", tmp);
    redundant_ = (tmp == 1);
  } else {
    redundant_ = false;
  }

  // Read cross section and threshold_idx data
  hid_t dset = open_dataset(group, "xs");

  // Get threshold index
  read_attribute(dset, "threshold_idx", xs_.threshold);

  // Read cross section values
  read_dataset(dset, xs_.value);
  close_dataset(dset);
  
  // Read products
  for (const auto& name : group_names(group)) {
    if (name.rfind("product_", 0) == 0) {
      hid_t pgroup = open_group(group, name.c_str());
      products_.emplace_back(pgroup);
      close_group(pgroup);
    }
  }
}

double PhotonuclearReaction::xs(int64_t i_grid, double interp_factor) const
{
  // If energy is below threshold, return 0. Otherwise interpolate between
  // nearest grid points
  return (i_grid < xs_.threshold)
           ? 0.0
           : (1.0 - interp_factor) * xs_.value[i_grid - xs_.threshold] +
               interp_factor * xs_.value[i_grid - xs_.threshold + 1];
}

double PhotonuclearReaction::xs(const PhotonuclearMicroXS& micro) const
{
  return this->xs(micro.index_grid, micro.interp_factor);
}

double PhotonuclearReaction::collapse_rate(span<const double> energy,
  span<const double> flux, const vector<double>& grid) const
{
  // Find index corresponding to first energy
  const auto& xs = xs_.value;
  int i_low = lower_bound_index(grid.cbegin(), grid.cend(), energy.front());

  // Check for threshold and adjust starting point if necessary
  int j_start = 0;
  int i_threshold = xs_.threshold;
  if (i_low < i_threshold) {
    i_low = i_threshold;
    while (energy[j_start + 1] < grid[i_low]) {
      ++j_start;
      if (j_start + 1 == energy.size())
        return 0.0;
    }
  }

  double xs_flux_sum = 0.0;

  for (int j = j_start; j < flux.size(); ++j) {
    double E_group_low = energy[j];
    double E_group_high = energy[j + 1];
    double flux_per_eV = flux[j] / (E_group_high - E_group_low);

    // Determine energy grid index corresponding to group high
    int i_high = i_low;
    while (grid[i_high + 1] < E_group_high && i_high + 1 < grid.size() - 1)
      ++i_high;

    // Loop over energy grid points within [E_group_low, E_group_high]
    for (; i_low <= i_high; ++i_low) {
      // Determine bounding grid energies and cross sections
      double E_l = grid[i_low];
      double E_r = grid[i_low + 1];
      if (E_l == E_r)
        continue;

      double xs_l = xs[i_low - i_threshold];
      double xs_r = xs[i_low + 1 - i_threshold];

      // Determine actual energies
      double E_low = std::max(E_group_low, E_l);
      double E_high = std::min(E_group_high, E_r);

      // Determine average cross section across segment
      double m = (xs_r - xs_l) / (E_r - E_l);
      double xs_low = xs_l + m * (E_low - E_l);
      double xs_high = xs_l + m * (E_high - E_l);
      double xs_avg = 0.5 * (xs_low + xs_high);

      // Add contribution from segment
      double dE = (E_high - E_low);
      xs_flux_sum += flux_per_eV * xs_avg * dE;
    }

    i_low = i_high;

    // Check for end of energy grid
    if (i_low + 1 == grid.size())
      break;
  }

  return xs_flux_sum;
}


//==============================================================================
// PhotonuclearInteraction implementation
//==============================================================================
int PhotonuclearInteraction::XS_TOTAL {0};
int PhotonuclearInteraction::XS_HEATING {1};
int PhotonuclearInteraction::XS_NEUTRON_PROD {2};

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
  
  // Read reactions
  for (auto name : group_names(rxs_group)) {
    if (starts_with(name, "reaction_")) {
      hid_t rx_group = open_group(rxs_group, name.c_str());
      reactions_.push_back(
        make_unique<PhotonuclearReaction>(rx_group, name_));
      close_group(rx_group);
    }
  }
  close_group(rxs_group);
  this->create_derived();
}

void PhotonuclearInteraction::create_derived()
{ 
  // Allocate and initialize cross section
  this->xs_ = xt::xtensor<double, 2>({energy_.size(),3}, 0.0);
  
  for (int i = 0; i < reactions_.size(); ++i) {
    
    const auto& rx {reactions_[i]};
    int n = rx->xs_.value.size();
    int j = rx->xs_.threshold;
    auto xs = xt::adapt(rx->xs_.value);
    auto nprod = xt::view(xs_, xt::range(j,j+n), XS_NEUTRON_PROD);
    for (const auto& p : rx->products_) {
      if (p.particle_ == ParticleType::neutron) {
        for (int k = 0; k < n; ++k) {
          double E = energy_[k];
          nprod[k] += xs[k] * (*p.yield_)(E);
        }
      }
    }
    if (rx->mt_==301) {
      auto heating = xt::view(xs_, xt::range(j,j+n), XS_HEATING);
      heating += xs;
    }
    // Skip redundant reactions
    if (rx->redundant_)
      continue;
    
    // Add contribution to total cross section
    auto total = xt::view(xs_, xt::range(j,j+n), XS_TOTAL);
    total += xs;  
  }
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
  if (E <= energy_[0]) {
    auto& xs {p.photonuclear_xs(index_)};
    xs.index_grid = -1;
    xs.heating = 0.0;
    xs.total = 0.0;
    xs.neutron_prod = 0.0;
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

  // Calculate microscopic total cross section
  xs.total = (1 - f) * xs_(i_grid, XS_TOTAL) + f * xs_(i_grid + 1, XS_TOTAL);

  // Calculate microscopic heating cross section
  xs.heating = (1 - f) * xs_(i_grid, XS_HEATING) + f * xs_(i_grid + 1, XS_HEATING);
  
  // Calculate microscopic nuclide neutron production cross section
  xs.neutron_prod = (1 - f) * xs_(i_grid, XS_NEUTRON_PROD) + f * xs_(i_grid + 1, XS_NEUTRON_PROD);

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
