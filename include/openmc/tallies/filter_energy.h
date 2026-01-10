#ifndef OPENMC_TALLIES_FILTER_ENERGY_H
#define OPENMC_TALLIES_FILTER_ENERGY_H

#include "openmc/array.h"
#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Bins the incident neutron energy.
//==============================================================================

class EnergyFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~EnergyFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "energy"; }
  FilterType type() const override { return FilterType::ENERGY; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<double>& bins() const { return bins_; }
  void set_bins(span<const double> bins);

  bool matches_transport_groups() const { return matches_transport_groups_; }

protected:
  //----------------------------------------------------------------------------
  // Data members

  vector<double> bins_;

  //! True if transport group number can be used directly to get bin number
  bool matches_transport_groups_ {false};
};

//==============================================================================
//! Bins the incident particle energy with gaussian broadening.
//==============================================================================

class GaussianBroadenedEnergyFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~GaussianBroadenedEnergyFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "gaussianbroadenedenergy"; }
  FilterType type() const override
  {
    return FilterType::GAUSSIAN_BROADENED_ENERGY;
  }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<double>& bins() const { return bins_; }
  double fwhm(double energy) const
  {
    return a_ + b_ * std::sqrt(energy * (1.0 + c_ * energy));
  }
  void set_bins(span<const double> bins);

protected:
  //----------------------------------------------------------------------------
  // Data members

  vector<double> bins_;
  double a_;
  double b_;
  double c_;
  vector<array<int64_t, 2>> offsets_;
  static constexpr double cutoff_ {std::numeric_limits<double>::denorm_min()};
};

//==============================================================================
//! Bins the outgoing neutron energy.
//!
//! Only scattering events use the get_all_bins functionality.  Nu-fission
//! tallies manually iterate over the filter bins.
//==============================================================================

class EnergyoutFilter : public EnergyFilter {
public:
  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "energyout"; }
  FilterType type() const override { return FilterType::ENERGY_OUT; }

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  std::string text_label(int bin) const override;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_ENERGY_H
