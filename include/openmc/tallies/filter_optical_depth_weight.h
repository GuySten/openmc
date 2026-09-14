#ifndef OPENMC_TALLIES_FILTER_OPTICAL_DEPTH_WEIGHT_H
#define OPENMC_TALLIES_FILTER_OPTICAL_DEPTH_WEIGHT_H

#include "openmc/tallies/filter.h"

namespace openmc {

//! Which attenuation coefficient the weight is measured with
enum class OpticalDepthBasis {
  TOTAL,       //!< the transport total, coherent scattering included
  NO_COHERENT  //!< the total less coherent scattering
};

//==============================================================================
//! Weights a next-event contribution by the optical depth it flew through.
//!
//! One bin, and it weights rather than selects -- hence the name, which says
//! what it does to a contribution rather than what it bins it by.  A filter
//! that bins by optical depth is a different thing and is free to take the
//! plainer name.  Weighting through a filter is not a stretch of the
//! mechanism: filter bins carry weights, and PointFilter applies
//! exp(-tau)/R^2 through exactly this route.  A tally carrying this
//! filter scores sum_i c_i tau_i, where c_i is what the contribution would
//! have scored on its own.  Divided by the same tally without the filter,
//! that is the flux-weighted mean optical depth: one number, not a
//! distribution.
//!
//! It exists for point-kernel buildup with a source that is not a point.  A
//! buildup factor is a function of the depth traversed, and a distributed
//! source offers no single one; recovering a depth from the flux instead
//! assumes a point source and reads an extended source's 1/R^2 spread as
//! attenuation, which is 8.5% wrong for a source four times the standoff
//! distance.  Folding the mean depth is 0.4% over the same range, so the mean
//! is enough and the distribution is not needed.
//!
//! It is also the only way to get the depth when the geometry contains voids,
//! where tau is not the attenuation coefficient times the distance and no
//! amount of knowing the source shape will supply it.
//!
//! Only a next-event flight has an optical depth, so the filter is inert
//! under any other estimator -- the rule PointFilter follows, and for the
//! same reason.
//==============================================================================

class OpticalDepthWeightFilter : public Filter {
public:
  ~OpticalDepthWeightFilter() = default;

  std::string type_str() const override { return "opticaldepthweight"; }
  FilterType type() const override { return FilterType::OPTICAL_DEPTH_WEIGHT; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  OpticalDepthBasis basis() const { return basis_; }
  void set_basis(OpticalDepthBasis basis) { basis_ = basis; }

private:
  //! Default TOTAL: the depth an actual photon attenuates on. NO_COHERENT is
  //! for reading a buildup factor written in the classic point-kernel
  //! convention, which excludes coherent scattering from the attenuation
  //! coefficient and folds it into the buildup factor instead. It changes
  //! only this weight -- PointFilter's exp(-tau) stays on the physical total.
  OpticalDepthBasis basis_ {OpticalDepthBasis::TOTAL};
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_OPTICAL_DEPTH_WEIGHT_H
