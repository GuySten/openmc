#include "openmc/photonuclear.h"

#include "openmc/array.h"
#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"
#include "openmc/distribution_multi.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/settings.h"
#include "openmc/string_utils.h"

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
double photonuclear_energy_max;

} // namespace data

//==============================================================================
// PhotonuclearReaction implementation
//==============================================================================

PhotonuclearReaction::PhotonuclearReaction(
  hid_t group, const std::string& nuclide_name, int64_t n_energy)
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

  // xs() and create_derived() both index relative to the threshold and neither
  // bounds-checks the top, so a file whose cross section does not exactly fill
  // the grid from its threshold upwards would read and write past the end of
  // the cross-section tensor.
  if (xs_.threshold < 0 ||
      xs_.threshold + static_cast<int64_t>(xs_.value.size()) != n_energy) {
    fatal_error(fmt::format(
      "Photonuclear MT={} of {} has a cross section of {} points starting at "
      "grid index {}, which does not fill the {}-point energy grid.",
      mt_, nuclide_name, xs_.value.size(), xs_.threshold, n_energy));
  }

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

//==============================================================================
// PhotonuclearInteraction implementation
//==============================================================================
int PhotonuclearInteraction::XS_TOTAL {0};
int PhotonuclearInteraction::XS_NEUTRON_PROD {1};

PhotonuclearInteraction::PhotonuclearInteraction(hid_t group)
{
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
        make_unique<PhotonuclearReaction>(rx_group, name_, energy_.size()));
      close_group(rx_group);
    }
  }
  close_group(rxs_group);

  // Identify the photofission reaction and its delayed precursor groups. The
  // ENDF convention is that products_[0] carries the prompt (or, when no
  // delayed data exists, the total) yield and products_[1..n] are the delayed
  // precursor groups.
  for (const auto& rx : reactions_) {
    if (rx->mt_ == N_FISSION || rx->mt_ == N_F || rx->mt_ == N_NF ||
        rx->mt_ == N_2NF || rx->mt_ == N_3NF) {
      if (rx->mt_ == N_FISSION || fission_rx_ == nullptr) {
        fission_rx_ = rx.get();
      }
      fissionable_ = true;
    }
  }
  if (fissionable_ && fission_rx_ != nullptr) {
    // nu() reads the prompt and total yields off products_[0] and treats
    // products_[1..n] as the delayed groups, so that ordering is a hard
    // requirement. The writer only preserves whatever order the ACE
    // secondary-particle blocks came in, and a photon product first would
    // make every photofission event bank a photon multiplicity as neutrons.
    // Check it here, where it can still be reported against the file.
    if (fission_rx_->products_.empty()) {
      fatal_error(
        fmt::format("Photofission reaction MT={} of {} has no products.",
          fission_rx_->mt_, name_));
    }
    const auto& first = fission_rx_->products_[0];
    if (!first.particle_.is_neutron() ||
        first.emission_mode_ == ReactionProduct::EmissionMode::delayed) {
      fatal_error(fmt::format(
        "The first product of photofission reaction MT={} of {} must be the "
        "prompt (or total) neutron yield, but it is a {} product of type {}.",
        fission_rx_->mt_, name_,
        first.emission_mode_ == ReactionProduct::EmissionMode::delayed
          ? "delayed"
          : "prompt",
        first.particle_.str()));
    }

    for (const auto& product : fission_rx_->products_) {
      if (product.particle_.is_neutron() &&
          product.emission_mode_ == ReactionProduct::EmissionMode::delayed) {
        ++n_precursor_;
      }
    }

    // The photonuclear ACE format carries no delayed neutron blocks, so data
    // processed through that route yields prompt emission only. The total
    // neutron count is still correct, but every neutron is born at the time of
    // the collision. Only time-dependent results are affected.
    if (n_precursor_ == 0 && settings::create_delayed_neutrons) {
      warning(fmt::format(
        "Photofission data for {} contains no delayed neutron precursor "
        "groups, so all photofission neutrons are emitted promptly. Results "
        "that depend on neutron emission time will be incorrect.",
        name_));
    }
  }

  // Total photofission neutron yield, when given separately from the prompt
  // yield
  if (object_exists(group, "total_nu")) {
    hid_t nu_group = open_group(group, "total_nu");
    total_nu_ = read_function(nu_group, "yield");
    close_group(nu_group);
  }

  // Recoverable fission energy release. This excludes the neutrino energy,
  // which escapes the system entirely, and is therefore the correct quantity
  // to use in place of the raw MT=18 Q value when scoring heating.
  if (object_exists(group, "fission_energy_release")) {
    hid_t fer_group = open_group(group, "fission_energy_release");
    fission_q_recov_ = read_function(fer_group, "q_recoverable");

    // Needed to scale the prompt photofission photon yield so that delayed
    // photons are accounted for
    prompt_photons_ = read_function(fer_group, "prompt_photons");
    delayed_photons_ = read_function(fer_group, "delayed_photons");
    close_group(fer_group);
  }

  this->create_derived();
}

void PhotonuclearInteraction::create_derived()
{
  // Allocate and initialize cross section
  this->xs_ = tensor::zeros<double>({energy_.size(), 2});

  for (int i = 0; i < reactions_.size(); ++i) {

    const auto& rx {reactions_[i]};
    int n = rx->xs_.value.size();
    int j = rx->xs_.threshold;
    auto xs = tensor::Tensor<double>(rx->xs_.value.data(), n);

    // Skip redundant reactions. This has to come before the neutron-production
    // loop as well: a redundant reaction that carries neutron products would
    // otherwise be counted on top of its own components, and
    // emit_forced_photoneutron() weights the forced neutron by
    // neutron_prod/total.
    if (rx->redundant_)
      continue;

    // NOTE: a reaction with multiple neutron products contributes its cross
    // section once per product here, and the yield is deliberately not folded
    // in. sample_photoneutron_product() enumerates products the same way, so
    // the per-product weighting cancels and the expected neutron production of
    // the forced emission in emit_forced_photoneutron() matches the analog sum
    // over products. Do not change one side without the other.
    for (const auto& p : rx->products_) {
      if (p.particle_ == ParticleType::neutron()) {
        for (int k = 0; k < n; ++k) {
          double E = energy_[k + j];
          if ((*p.yield_)(E) > 0.0)
            xs_(j + k, XS_NEUTRON_PROD) += xs[k];
        }
      }
    }

    // Add contribution to total cross section
    xs_.slice(tensor::range(j, j + n), XS_TOTAL) += xs;
  }
}

PhotonuclearInteraction::~PhotonuclearInteraction()
{
  data::photonuclear_map.erase(name_);
}

double PhotonuclearInteraction::nu(double E, EmissionMode mode, int group) const
{
  if (!fissionable_ || fission_rx_ == nullptr)
    return 0.0;

  switch (mode) {
  case EmissionMode::prompt:
    return (*fission_rx_->products_[0].yield_)(E);
  case EmissionMode::delayed:
    if (n_precursor_ > 0 && settings::create_delayed_neutrons) {
      if (group >= 1 && group < fission_rx_->products_.size()) {
        return (*fission_rx_->products_[group].yield_)(E);
      }
      double nu {0.0};
      for (int i = 1; i < fission_rx_->products_.size(); ++i) {
        const auto& product = fission_rx_->products_[i];
        if (!product.particle_.is_neutron())
          continue;
        if (product.emission_mode_ == ReactionProduct::EmissionMode::delayed) {
          nu += (*product.yield_)(E);
        }
      }
      return nu;
    }
    return 0.0;
  case EmissionMode::total:
    if (total_nu_ && settings::create_delayed_neutrons) {
      return (*total_nu_)(E);
    }
    return (*fission_rx_->products_[0].yield_)(E);
  }
  UNREACHABLE();
}

double PhotonuclearInteraction::fission_q_recoverable(double E) const
{
  return fission_q_recov_ ? (*fission_q_recov_)(E) : 0.0;
}

double PhotonuclearInteraction::delayed_photon_factor(double E) const
{
  if (!settings::delayed_photon_scaling)
    return 1.0;
  if (!prompt_photons_ || !delayed_photons_)
    return 1.0;

  double energy_prompt = (*prompt_photons_)(E);
  if (energy_prompt <= 0.0)
    return 1.0;

  double energy_delayed = (*delayed_photons_)(E);
  return (energy_prompt + energy_delayed) / energy_prompt;
}

void PhotonuclearInteraction::calculate_xs(Particle& p) const
{
  // Perform binary search on the element energy grid in order to determine
  // which points to interpolate between
  int n_grid = energy_.size();
  double E = p.E();
  int i_grid;
  if (E <= energy_[0] || E > energy_(n_grid - 1)) {
    auto& xs {p.photonuclear_xs(index_)};
    xs.index_grid = -1;
    xs.total = 0.0;
    xs.neutron_prod = 0.0;
    xs.last_E = p.E();
    return;
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

  // Calculate microscopic nuclide neutron production cross section
  xs.neutron_prod = (1 - f) * xs_(i_grid, XS_NEUTRON_PROD) +
                    f * xs_(i_grid + 1, XS_NEUTRON_PROD);

  xs.last_E = p.E();
}

//==============================================================================
// Non-member functions
//==============================================================================

void free_memory_photonuclear()
{
  data::photonuclears.clear();
}

//! Maximum laboratory energy of a photoneutron from a given reaction at a
//! given incident photon energy.
//!
//! The outgoing distributions are tabulated in whichever frame the reaction
//! declares. For a center-of-mass reaction the tabulated maximum is a
//! center-of-mass energy and has to be transformed; the extremum over the
//! emission cosine is at mu_cm = +1, i.e. forward emission. The transformation
//! is the massless-projectile one used in emit_photonuclear_product(): the
//! photon carries momentum E/c, so the center-of-mass speed is E/(A*m_n*c).
//! For a laboratory-frame reaction the tabulated maximum is already the
//! laboratory maximum and is returned unchanged.
static double max_photoneutron_energy_lab(const PhotonuclearReaction& rx,
  const ReactionProduct& product, double awr, double E_in)
{
  double E_out = product.max_energy(E_in);
  if (E_out <= 0.0)
    return 0.0;

  if (!rx.scatter_in_cm_)
    return E_out;

  // Forward emission in the center-of-mass frame, mu_cm = +1
  double A = awr;
  return E_out + (E_in / A) * std::sqrt(2.0 * E_out / MASS_NEUTRON_EV) +
         (E_in * E_in) / (2.0 * MASS_NEUTRON_EV * A * A);
}

double max_safe_photon_energy(
  double E_max_neutron, std::string& limiting_nuclide, int& limiting_mt)
{
  double E_safe = INFTY;
  limiting_nuclide.clear();
  limiting_mt = 0;

  for (const auto& nuc : data::photonuclears) {
    if (nuc->energy_.size() < 2)
      continue;

    for (const auto& rx : nuc->reactions_) {
      if (rx->redundant_)
        continue;

      for (const auto& product : rx->products_) {
        if (product.particle_ != ParticleType::neutron())
          continue;

        // Walk the reaction's own energy grid rather than assuming the bound
        // is monotonic in E_in -- for tabulated laws it need not be. The first
        // grid point that exceeds the neutron data range brackets the limit.
        //
        // There is deliberately no "is the top of the grid safe?" shortcut
        // here. It would assume exactly the monotonicity this walk exists to
        // avoid assuming: a reaction can be safe at the top of the grid and
        // unsafe in the middle, and skipping it then leaves the photon ceiling
        // too high, which surfaces as a fatal_error mid-transport.
        int i_hi = -1;
        for (int i = rx->xs_.threshold; i < nuc->energy_.size(); ++i) {
          double E_in = nuc->energy_[i];
          if ((*product.yield_)(E_in) <= 0.0)
            continue;
          if (max_photoneutron_energy_lab(*rx, product, nuc->awr_, E_in) >
              E_max_neutron) {
            i_hi = i;
            break;
          }
        }
        if (i_hi < 0)
          continue;

        // The very first energy at which this product appears already exceeds
        // the neutron data range, so there is no safe window at all. Record it
        // the same way as any other limit, so that the reaction reported is the
        // one that actually set E_safe.
        if (i_hi == 0) {
          if (0.0 < E_safe) {
            E_safe = 0.0;
            limiting_nuclide = nuc->name_;
            limiting_mt = rx->mt_;
          }
          continue;
        }

        // Bisect within the bracketing bin for a tighter limit
        double lo = nuc->energy_[i_hi - 1];
        double hi = nuc->energy_[i_hi];
        for (int it = 0; it < 60; ++it) {
          double mid = 0.5 * (lo + hi);
          if (max_photoneutron_energy_lab(*rx, product, nuc->awr_, mid) >
              E_max_neutron) {
            hi = mid;
          } else {
            lo = mid;
          }
        }

        if (lo < E_safe) {
          E_safe = lo;
          limiting_nuclide = nuc->name_;
          limiting_mt = rx->mt_;
        }
      }
    }
  }

  return E_safe;
}

} // namespace openmc
