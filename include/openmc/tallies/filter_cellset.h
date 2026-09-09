#ifndef OPENMC_TALLIES_FILTER_CELL_SET_H
#define OPENMC_TALLIES_FILTER_CELL_SET_H

#include <cstdint>
#include <string>
#include <unordered_set>

#include "openmc/span.h"
#include "openmc/tallies/filter.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Which side of a region's boundary a crossing goes.
//==============================================================================

enum class CellSetSense {
  NET, //!< Signed: +1 leaving the region, -1 entering it
  OUT, //!< Crossings leaving the region only
  IN   //!< Crossings entering the region only
};

//==============================================================================
//! Bins surface crossings by the region boundary they cross.
//!
//! A region is a set of cells treated as a single unit. A crossing matches a
//! region only when it has one end inside and one end outside, so crossings
//! internal to a region are not boundary crossings and do not score. The cells
//! of a region need not be adjacent.
//!
//! Because the direction of a crossing follows from the cells on either side
//! of it, currents binned by this filter do not depend on the orientation of
//! the surface that was crossed. The NET sense carries its sign as a filter
//! weight so that a net current accumulates in a single bin, which keeps its
//! variance correct; a net current recovered by subtracting an OUT bin from an
//! IN bin would not, since the two are correlated and no covariance between
//! bins is tracked.
//!
//! At a boundary condition the cells either side of a crossing no longer say
//! what happened, so the boundary decides instead: leaking out of the model
//! counts as leaving the region, and a reflective, white or periodic boundary
//! counts as one crossing out and one back in.
//!
//! Bins are region-major: bin = i_region * n_senses + i_sense.
//==============================================================================

class CellSetFilter : public Filter {
public:
  //----------------------------------------------------------------------------
  // Constructors, destructors

  ~CellSetFilter() = default;

  //----------------------------------------------------------------------------
  // Methods

  std::string type_str() const override { return "cellset"; }
  FilterType type() const override { return FilterType::CELL_SET; }

  void from_xml(pugi::xml_node node) override;

  void get_all_bins(const Particle& p, TallyEstimator estimator,
    FilterMatch& match) const override;

  void to_statepoint(hid_t filter_group) const override;

  std::string text_label(int bin) const override;

  //----------------------------------------------------------------------------
  // Accessors

  //! Set the regions from a flattened list of cell indices
  //!
  //! \param[in] cells Cell indices for every region, concatenated
  //! \param[in] region_sizes Number of cells in each region; must sum to the
  //!   size of \c cells
  void set_regions(span<int32_t> cells, span<int32_t> region_sizes);

  //! Set which senses are binned, in the order their bins appear
  void set_senses(const vector<CellSetSense>& senses);

  int n_regions() const { return regions_.size(); }
  const vector<CellSetSense>& senses() const { return senses_; }

private:
  //----------------------------------------------------------------------------
  // Methods

  //! Is any coordinate level of the pre-crossing position inside a region?
  bool was_inside(const Particle& p, int i_region) const;

  //! Is any coordinate level of the post-crossing position inside a region?
  bool is_inside(const Particle& p, int i_region) const;

  //! Push the bins for one region, given which way the crossing went
  void match_region(FilterMatch& match, int i_region, bool leaving) const;

  //----------------------------------------------------------------------------
  // Data members

  //! Cell indices making up each region, in the order they were given. Kept
  //! alongside the lookup sets so that statepoint output and bin labels do not
  //! depend on hash ordering.
  vector<vector<int32_t>> region_cells_;

  //! Cell indices making up each region. A set per region rather than one map
  //! from cell to region so that regions may overlap.
  vector<std::unordered_set<int32_t>> regions_;

  //! Senses binned for each region, in bin order
  vector<CellSetSense> senses_ {CellSetSense::NET};
};

//==============================================================================
// Non-member functions
//==============================================================================

//! Convert a sense to the string used in XML and statepoint files
std::string cell_set_sense_str(CellSetSense sense);

//! Convert an XML string to a sense, aborting on an unrecognized value
CellSetSense cell_set_sense_from_str(const std::string& str);

} // namespace openmc
#endif // OPENMC_TALLIES_FILTER_CELL_SET_H
