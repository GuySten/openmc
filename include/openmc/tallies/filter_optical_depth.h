#ifndef OPENMC_TALLIES_FILTER_OPTICAL_DEPTH_H
#define OPENMC_TALLIES_FILTER_OPTICAL_DEPTH_H

#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Bins a next-event contribution by the optical depth it flew through.
//!
//! The next-event estimator already measures, for every contribution, the
//! number of mean free paths between the emitting event and the detector: it
//! is what the attenuation factor exp(-tau) applied by PointFilter is built
//! from. This filter exposes that same quantity as a binning dimension.
//!
//! The motivation is point-kernel buildup. A buildup factor B is a function of
//! the optical depth traversed, and it is strongly non-linear in it, so
//! collapsing contributions of different depth into one number and applying
//! B(<tau>) afterwards is not the same as applying B(tau) to each. Resolving
//! tally in tau lets the fold be done per depth, and the bin width then
//! controls the error rather than the spread of the source region.
//!
//! Only the next-event estimator produces the quantity, so the filter is inert
//! under any other estimator -- the same rule PointFilter follows, and for the
//! same reason.
//==============================================================================

class OpticalDepthFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~OpticalDepthFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "opticaldepth"; }
  FilterType type() const override { return FilterType::OPTICAL_DEPTH; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  void set_bins(span<double> bins);

  const vector<double>& bins() const { return bins_; }

protected:
  //----------------------------------------------------------------------------
  // Data members

  //! Bin edges in mean free paths, monotonically increasing
  vector<double> bins_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_OPTICAL_DEPTH_H
