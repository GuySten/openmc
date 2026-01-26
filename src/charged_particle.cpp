#include "openmc/charged_particle.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <fmt/core.h>

#include "openmc/hdf5_interface.h"
#include "openmc/search.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace data {
std::unordered_map<std::string, int> charged_particle_map;
vector<unique_ptr<IncidentChargedParticle>> charged_particles;
} // namespace data

//==============================================================================
// IncidentChargedParticle implementation
//==============================================================================

IncidentChargedParticle::IncidentChargedParticle(hid_t group)
{
  // Read attributes
  read_attribute(group, "projectile", projectile_);
  read_attribute(group, "target", target_);
  read_attribute(group, "Z", Z_);
  read_attribute(group, "A", A_);
  read_attribute(group, "metastable", metastable_);
  read_attribute(group, "atomic_weight_ratio", awr_);

  // Construct full name
  name_ = projectile_ + "_" + target_;

  // Read energy grid
  read_dataset(group, "energy", energy_);
  n_energy_ = energy_.size();

  // Read MT numbers
  read_dataset(group, "MT", MT_);
  n_reactions_ = MT_.size();

  // Read Q-values
  read_dataset(group, "Q_value", Q_value_);

  // Read ejectile particle types (stored as integers matching ParticleType enum)
  vector<int> ejectile_indices;
  read_dataset(group, "ejectile", ejectile_indices);
  ejectile_.resize(n_reactions_);
  for (int j = 0; j < n_reactions_; ++j) {
    ejectile_[j] = static_cast<ParticleType>(ejectile_indices[j]);
  }

  // Read cross section matrix (stored as n_energy x n_reactions)
  xt::xtensor<double, 2> xs_matrix;
  read_dataset(group, "xs", xs_matrix);

  // Flatten to 1D vector for efficient access
  xs_.resize(n_energy_ * n_reactions_);
  for (int i = 0; i < n_energy_; ++i) {
    for (int j = 0; j < n_reactions_; ++j) {
      xs_[i * n_reactions_ + j] = xs_matrix(i, j);
    }
  }

  // Read reaction products if available (for energy-dependent yields)
  products_.resize(n_reactions_);
  if (object_exists(group, "reactions")) {
    hid_t rxs_group = open_group(group, "reactions");

    // Iterate over reactions
    for (int j = 0; j < n_reactions_; ++j) {
      int mt = MT_[j];
      std::string rx_name = fmt::format("reaction_{:03d}", mt);

      if (object_exists(rxs_group, rx_name.c_str())) {
        hid_t rx_group = open_group(rxs_group, rx_name.c_str());

        // Count number of products
        int n_products = 0;
        for (int i = 0; ; ++i) {
          std::string prod_name = fmt::format("product_{}", i);
          if (!object_exists(rx_group, prod_name.c_str()))
            break;
          ++n_products;
        }

        // Read each product
        for (int i = 0; i < n_products; ++i) {
          std::string prod_name = fmt::format("product_{}", i);
          hid_t prod_group = open_group(rx_group, prod_name.c_str());
          products_[j].emplace_back(prod_group);
          close_group(prod_group);
        }

        close_group(rx_group);
      }
    }

    close_group(rxs_group);
  }
}

void IncidentChargedParticle::find_energy(double E, int& i, double& f) const
{
  // Handle boundary cases
  if (E <= energy_.front()) {
    i = 0;
    f = 0.0;
    return;
  }
  if (E >= energy_.back()) {
    i = n_energy_ - 2;
    f = 1.0;
    return;
  }

  // Binary search for energy interval
  i = lower_bound_index(energy_.begin(), energy_.end(), E);
  if (i > 0)
    --i;

  // Calculate interpolation factor (log-log interpolation)
  double E1 = energy_[i];
  double E2 = energy_[i + 1];
  f = std::log(E / E1) / std::log(E2 / E1);
}

double IncidentChargedParticle::xs(int i_reaction, double E) const
{
  // Find energy index and interpolation factor
  int i;
  double f;
  find_energy(E, i, f);

  // Get cross sections at bounding energies
  double xs1 = xs_[i * n_reactions_ + i_reaction];
  double xs2 = xs_[(i + 1) * n_reactions_ + i_reaction];

  // Log-log interpolation (handles Gamow-like behavior better)
  if (xs1 > 0.0 && xs2 > 0.0) {
    return xs1 * std::exp(f * std::log(xs2 / xs1));
  } else if (xs1 > 0.0) {
    return xs1 * (1.0 - f);
  } else if (xs2 > 0.0) {
    return xs2 * f;
  } else {
    return 0.0;
  }
}

double IncidentChargedParticle::total_xs(double E) const
{
  double total = 0.0;
  for (int j = 0; j < n_reactions_; ++j) {
    total += xs(j, E);
  }
  return total;
}

//==============================================================================
// Non-member functions
//==============================================================================

void read_charged_particle_data(const std::string& filename)
{
  // Open HDF5 file
  hid_t file_id = file_open(filename, 'r');

  // Iterate over all groups (each is a charged particle dataset)
  // For simplicity, we expect one group per file matching the pattern
  // {projectile}_{target}
  hsize_t num_objects;
  H5Gget_num_objs(file_id, &num_objects);

  for (hsize_t i = 0; i < num_objects; ++i) {
    // Get name of object
    char name[256];
    H5Gget_objname_by_idx(file_id, i, name, sizeof(name));

    // Check if it's a group
    H5G_stat_t statbuf;
    H5Gget_objinfo(file_id, name, 0, &statbuf);
    if (statbuf.type != H5G_GROUP)
      continue;

    // Open group and create IncidentChargedParticle
    hid_t group_id = open_group(file_id, name);
    auto cp = make_unique<IncidentChargedParticle>(group_id);
    close_group(group_id);

    // Add to global data
    int index = data::charged_particles.size();
    data::charged_particle_map[cp->name_] = index;
    data::charged_particles.push_back(std::move(cp));
  }

  file_close(file_id);
}

void charged_particles_clear()
{
  data::charged_particles.clear();
  data::charged_particle_map.clear();
}

ParticleType particle_type_from_string(const std::string& name)
{
  if (name == "neutron")
    return ParticleType::neutron;
  if (name == "photon")
    return ParticleType::photon;
  if (name == "electron")
    return ParticleType::electron;
  if (name == "positron")
    return ParticleType::positron;
  if (name == "proton")
    return ParticleType::proton;
  if (name == "deuteron")
    return ParticleType::deuteron;
  if (name == "triton")
    return ParticleType::triton;
  if (name == "helion")
    return ParticleType::helion;
  if (name == "alpha")
    return ParticleType::alpha;

  throw std::invalid_argument("Unknown particle type: " + name);
}

std::string particle_type_to_string(ParticleType type)
{
  switch (type) {
  case ParticleType::neutron:
    return "neutron";
  case ParticleType::photon:
    return "photon";
  case ParticleType::electron:
    return "electron";
  case ParticleType::positron:
    return "positron";
  case ParticleType::proton:
    return "proton";
  case ParticleType::deuteron:
    return "deuteron";
  case ParticleType::triton:
    return "triton";
  case ParticleType::helion:
    return "helion";
  case ParticleType::alpha:
    return "alpha";
  default:
    return "unknown";
  }
}

} // namespace openmc
