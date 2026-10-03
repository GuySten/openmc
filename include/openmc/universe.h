#ifndef OPENMC_UNIVERSE_H
#define OPENMC_UNIVERSE_H

#include <algorithm> // for upper_bound

#include "openmc/bounding_box.h"
#include "openmc/cell.h"

namespace openmc {

#ifdef OPENMC_DAGMC_ENABLED
class DAGUniverse;
#endif

class GeometryState;
class Universe;
class UniversePartitioner;

namespace model {

extern std::unordered_map<int32_t, int32_t> universe_map;
extern vector<unique_ptr<Universe>> universes;

//! Distributed cell offsets of all fill cells and lattice tiles
extern vector<int32_t> distribcell_offsets;

} // namespace model

//==============================================================================
//! Distributed cell maps that can be reached below a universe or lattice.
//
//! Distributed cell offsets are only stored for the maps (universes containing
//! distributed cells) reachable below a fill cell or lattice tile. A layout
//! gives the position of each reachable map among those offsets. Maps are
//! numbered so that the maps reachable below a universe are mostly
//! consecutive, so the layout stores runs of consecutive map indices.
//==============================================================================

class DistribcellLayout {
public:
  //! Build the layout from a sorted list of map indices
  void build(const vector<int32_t>& maps);

  //! \return Position of a map among the offsets, or C_NONE if the map can't
  //!   be reached
  int32_t slot(int32_t map) const
  {
    // Most layouts have few runs, for which a linear search is fastest
    if (runs_.size() <= 8) {
      for (const auto& run : runs_) {
        auto d = static_cast<uint32_t>(map - run.first);
        if (d < static_cast<uint32_t>(run.n))
          return run.slot + d;
      }
      return C_NONE;
    }
    auto it = std::upper_bound(runs_.begin(), runs_.end(), map,
      [](int32_t m, const Run& run) { return m < run.first; });
    if (it == runs_.begin())
      return C_NONE;
    --it;
    auto d = static_cast<uint32_t>(map - it->first);
    return d < static_cast<uint32_t>(it->n) ? it->slot + d : C_NONE;
  }

  //! \return Number of maps that can be reached
  int32_t size() const { return size_; }

private:
  struct Run {
    int32_t first; //!< First map index of the run
    int32_t n;     //!< Number of consecutive map indices
    int32_t slot;  //!< Position of the first map among the offsets
  };
  vector<Run> runs_;
  int32_t size_ {0};
};

//==============================================================================
//! A geometry primitive that fills all space and contains cells.
//==============================================================================

class Universe {
public:
  int32_t id_;            //!< Unique ID
  vector<int32_t> cells_; //!< Cells within this universe
  int32_t n_instances_;   //!< Number of instances of this universe

  //! Distributed cell maps that can be reached in this universe
  DistribcellLayout distribcell_layout_;

  //! \brief Write universe information to an HDF5 group.
  //! \param group_id An HDF5 group id.
  virtual void to_hdf5(hid_t group_id) const;

  virtual bool find_cell(GeometryState& p) const;

  BoundingBox bounding_box() const;

  /* By default, universes are CSG universes. The DAGMC
   * universe overrides standard behaviors, and in the future,
   * other things might too.
   */
  virtual GeometryType geom_type() const { return GeometryType::CSG; }

  unique_ptr<UniversePartitioner> partitioner_;
};

//==============================================================================
//! Speeds up geometry searches by grouping cells in a search tree.
//
//! Currently this object only works with universes that are divided up by a
//! bunch of z-planes.  It could be generalized to other planes, cylinders,
//! and spheres.
//==============================================================================

class UniversePartitioner {
public:
  explicit UniversePartitioner(const Universe& univ);

  //! Return the list of cells that could contain the given coordinates.
  const vector<int32_t>& get_cells(Position r, Direction u) const;

private:
  //! A sorted vector of indices to surfaces that partition the universe
  vector<int32_t> surfs_;

  //! Vectors listing the indices of the cells that lie within each partition
  //
  //! There are n+1 partitions with n surfaces.  `partitions_.front()` gives the
  //! cells that lie on the negative side of `surfs_.front()`.
  //! `partitions_.back()` gives the cells that lie on the positive side of
  //! `surfs_.back()`.  Otherwise, `partitions_[i]` gives cells sandwiched
  //! between `surfs_[i-1]` and `surfs_[i]`.
  vector<vector<int32_t>> partitions_;
};

} // namespace openmc
#endif // OPENMC_UNIVERSE_H
