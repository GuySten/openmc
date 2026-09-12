#ifndef OPENMC_TALLIES_FILTER_POINT_H
#define OPENMC_TALLIES_FILTER_POINT_H

#include "openmc/position.h"
#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

class PointFilter;

//! Whether an exclusion sphere could be shown to hold a single material
enum class SphereCheck {
  SINGLE_MATERIAL,    //!< proven to hold one material throughout
  MULTIPLE_MATERIALS, //!< a cell of a different material reaches into it
  UNDECIDABLE,        //!< beyond what this test can establish either way
  OUTSIDE_MODEL       //!< the detector could not be located in the geometry
};

//! Test whether a detector's exclusion sphere provably holds one material.
//!
//! The sphere average assumes a single total cross section throughout, so what
//! matters is the material, not the cell: a sphere spilling across a boundary
//! into more of the same material is perfectly fine, and that is the common
//! case wherever cells subdivide a uniform region.
//!
//! Two stages, both exact. Leaving a cell means crossing one of the surfaces
//! of its region, so if every such surface, at every level of the coordinate
//! hierarchy, is farther away than the radius, the sphere cannot leave the
//! cell it starts in and a cell holds one material by construction. Failing
//! that, the sphere is still confined to the universe at its innermost level
//! provided no surface above that level is within reach; the cells of that
//! universe whose bounding box the sphere reaches are then the only ones it
//! can touch, and if they all carry the detector's own material the sphere
//! holds one material after all. Bounding boxes over-approximate, so this can
//! only ever consider too many cells, never too few.
//!
//! Nothing here is sampled, so SINGLE_MATERIAL is a proof. The converse is
//! weaker: a cell of another material whose bounding box is in reach may not
//! really intersect the sphere, so MULTIPLE_MATERIALS means "not proven safe,
//! and here is why". A lattice, whose element boundaries are not surfaces of
//! any cell's region, a surface with no closed-form distance to a point, a
//! neighbouring cell filled by a universe rather than a material, and a sphere
//! that escapes its innermost universe all give UNDECIDABLE rather than a
//! result that cannot be trusted.
//!
//! \param[in] pos Centre of the sphere
//! \param[in] r0 Radius of the sphere
//! \param[out] nearest Distance to the closest cell boundary found, left
//!   untouched if the detector could not be located
SphereCheck check_exclusion_sphere(Position pos, double r0, double& nearest);

//! Warn about every exclusion sphere of a filter not proven to hold one
//! material
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
