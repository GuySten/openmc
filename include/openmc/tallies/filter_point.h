#ifndef OPENMC_TALLIES_FILTER_POINT_H
#define OPENMC_TALLIES_FILTER_POINT_H

#include "openmc/position.h"
#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

class PointFilter;

//! Whether an exclusion sphere could be shown to stay within a single cell
enum class SphereCheck {
  CONFINED,     //!< proven to stay inside one cell
  NOT_CONFINED, //!< a bounding surface is closer than the radius
  UNDECIDABLE,  //!< a lattice, or a surface with no closed-form distance
  OUTSIDE_MODEL //!< the detector could not be located in the geometry
};

//! Test whether a detector's exclusion sphere provably stays in one cell.
//!
//! Leaving a cell means crossing one of the surfaces of its region, so if
//! every such surface is farther from the detector than the sphere's radius,
//! at every level of the coordinate hierarchy, the sphere cannot escape the
//! cell it starts in. The test uses closed-form point-to-surface distances
//! rather than sampled directions, so a CONFINED result is exact.
//!
//! It proves safety, not danger: a surface within reach may separate two cells
//! of the same material, which is harmless for the exclusion sphere. Lattice
//! element boundaries are not cell surfaces, and the general quadric and the
//! tori have no closed-form distance to a point, so either makes the answer
//! UNDECIDABLE rather than wrong.
//!
//! \param[in] pos Centre of the sphere
//! \param[in] r0 Radius of the sphere
//! \param[out] nearest Distance to the closest bounding surface found, left
//!   untouched unless the result is CONFINED or NOT_CONFINED
SphereCheck check_exclusion_sphere(Position pos, double r0, double& nearest);

//! Warn about every exclusion sphere of a filter that is not provably confined
void check_point_detector_spheres(const PointFilter& filt);

//==============================================================================
//! Bins tally by point detectors
//==============================================================================

class PointFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~PointFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "point"; }
  FilterType type() const override { return FilterType::POINT; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  const vector<std::pair<Position, double>>& detectors() const
  {
    return detectors_;
  }

  void set_detectors(span<std::pair<Position, double>> detectors);

  //! Record, for each of this filter's bins, which entry of
  //! model::active_point_detectors it sits on. Called once per batch from
  //! setup_active_tallies(), after that list has been assembled.
  void build_detector_bins();

private:
  //----------------------------------------------------------------------------
  // Data members

  vector<std::pair<Position, double>> detectors_;

  //! Parallel to detectors_: the index into model::active_point_detectors
  //! that each bin sits on, or C_NONE if this filter's tally is not active.
  //! Mapping this way rather than the reverse keeps the behaviour of a filter
  //! that puts two bins on one position -- different exclusion radii at the
  //! same point -- where a detector-to-bin map could only name one of them.
  //! Rebuilt every batch by build_detector_bins().
  vector<int> bin_detector_;
};

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_POINT_H
