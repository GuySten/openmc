#ifndef OPENMC_VOLUME_OCTREE_H
#define OPENMC_VOLUME_OCTREE_H

//! \file volume_octree.h
//! \brief Deterministic CSG volume calculation by octree domain decomposition.
//!
//! Bracketing form of Millman/Griesheimer/Nease/Snoeyink (PHYSOR 2012). Each
//! octree box is classified against every surface in the regions of interest as
//! wholly negative, wholly positive, or straddling; the region expression is
//! evaluated in three-valued logic. A TRUE box contributes its volume to the
//! lower bound, a box that neither resolves nor yields to a closed-form
//! integrator contributes to the slack:
//!
//!     lower <= V <= lower + slack
//!
//! No statistical content. Every classifier is conservative -- it may report
//! BOTH where a tighter test would resolve, but never reports NEGATIVE or
//! POSITIVE wrongly. That soundness is verified directly in
//! tests/cpp_unit_tests/.

#include <atomic>
#include <cstdint>
#include <unordered_map>

#include "openmc/array.h"
#include "openmc/position.h"
#include "openmc/vector.h"
#include "openmc/volume_octree_math.h"

namespace openmc {

class Cell;
class Surface;
class Universe;

//! Extract a surface's quadratic form. False for tori (quartic) and for
//! surface types this does not recognise, e.g. DAGMC.
bool surface_quadric_form(const Surface& surf, QuadricForm& q);

//! Classify a box against a surface.
//!
//! Exact whenever the quadratic form is separable in the box axes: planes,
//! axis-aligned cylinders, spheres and cones against unrotated boxes, and
//! spheres against any box. Exact for tori against unrotated boxes. Falls back
//! to the centred interval form for coupled cross terms, which is sound and
//! O(h^2). Anything else returns BOTH, which forces refinement.
BoxSense classify_box(const Surface& surf, const OctBox& box);

//==============================================================================
//! Result for one domain.
//==============================================================================

struct VolumeBracket {
  double lower {0.0}; //!< volume proven to be inside the domain
  double slack {0.0}; //!< volume that refinement and the integrators left open

  double upper() const { return lower + slack; }
  double midpoint() const { return lower + 0.5 * slack; }
  double half_width() const { return 0.5 * slack; }
};

//==============================================================================
//! Driver
//==============================================================================

class OctreeVolumeCalculation {
public:
  enum class DomainType { CELL, MATERIAL, UNIVERSE };

  OctreeVolumeCalculation(DomainType type, const vector<int>& domain_ids,
    Position lower_left, Position upper_right);

  //! Deepen the octree until every bracket meets the tolerance or the depth
  //! guard is hit. Parallel over OpenMP threads and MPI ranks.
  void run();

  const vector<VolumeBracket>& results() const { return results_; }

  //! Nuclide atom counts for one domain, in the same form the stochastic path
  //! produces. The uncertainty entries are hard bounds propagated from the
  //! slack rather than standard deviations; both are monotone in the volume so
  //! they propagate identically, only the interpretation differs.
  void fill_nuclides(int i_domain, vector<int>& nuclides, vector<double>& atoms,
    vector<double>& uncertainty) const;

  double worst_half_width() const;

  //! Depth actually reached by the last sweep.
  int depth_reached() const { return depth_reached_; }

  double tolerance_ {1.0e-3}; //!< absolute half-width target, cm^3

  //! Depth guard. Deliberately generous: reaching a 1e-6 cm^3 half-width on a
  //! pincell by refinement alone needs depth ~25, and a guard of 20 silently
  //! capped the achievable answer short of any tight tolerance.
  int max_depth_ {30};

  //! Slice cap for the revolved-surface integrator. Spread falls as 1/N, so the
  //! count needed is predicted from a 32-slice probe; this is a wall-clock
  //! guard on that prediction, not an accuracy setting.
  int max_slices_ {1 << 26};

  int start_depth_ {6}; //!< first depth tried by the deepening loop

  //! Frontier boxes to aim for per worker. The frontier is grown adaptively --
  //! see build_frontier -- so a model that resolves at the root costs one box
  //! rather than 8^k.
  int tasks_per_worker_ {8};

private:
  //! One level of the descent, enough to reconstruct a distribcell instance.
  struct PathEntry {
    int32_t i_cell;        //!< the filled cell we descended through
    array<int, 3> lat_idx; //!< lattice element, when that cell held a lattice
    bool has_lat;
  };

  struct Node {
    OctBox box;
    int32_t i_univ;
    int depth;
    vector<PathEntry> path;
  };

  //! Dense per-thread accumulator.
  //!
  //! Dense rather than a map so the merge is a flat loop of atomic adds with no
  //! critical section, and so the MPI reduction is a plain Allreduce. The
  //! (domain, material) pairs are enumerated up front by the reachability pass
  //! below, which is the same pass that fixes slack charging.
  struct Acc {
    vector<double> lower, slack, mat_lower, mat_slack;
    void init(size_t n_slots, size_t n_pairs)
    {
      lower.assign(n_slots, 0.0);
      slack.assign(n_slots, 0.0);
      mat_lower.assign(n_pairs, 0.0);
      mat_slack.assign(n_pairs, 0.0);
    }
  };

  //! Pre-digested surface, so the hot loop does no dynamic_cast.
  struct SurfaceProxy {
    enum class Kind : uint8_t { QUADRIC, TORUS, OPAQUE };
    Kind kind {Kind::OPAQUE};
    QuadricForm q;
    int ax {0};
    Position tc;
    double A {0.}, B {0.}, C {0.};
  };

  //! Everything a slack charge beneath a universe has to touch, deduplicated.
  //!
  //! Walking the universe tree per slack event charged a lattice's element
  //! universe once per POSITION -- 289 times for a 17x17 assembly -- which
  //! inflated slack by that factor and made the tolerance unreachable, besides
  //! costing O(n_elements) per event. Computed once here instead.
  struct Reach {
    vector<int> slots;
    vector<int> pairs;
    bool done {false};
  };

  BoxSense classify(int32_t i_surf, const OctBox& box) const;

  //! Handle one box: credit it, integrate it, charge it, or emit the children
  //! it needs subdivided into. `box`, `i_univ` and `path` are updated in place
  //! when the box descends through a fill without subdividing.
  int process(OctBox& box, int32_t& i_univ, int depth, int max_depth,
    vector<PathEntry>& path, Acc& acc, OctBox* kids) const;

  void descend(OctBox box, int32_t i_univ, int depth, int max_depth,
    vector<PathEntry>& path, Acc& acc) const;

  //! Grow a work list by processing nodes breadth-first until there is enough
  //! for the available workers, or until nothing is left to subdivide.
  void build_frontier(int max_depth, Acc& serial, vector<Node>& frontier) const;

  void sweep(int max_depth);

  bool try_integrate(const OctBox& box, const Universe& u,
    const std::unordered_map<int32_t, BoxSense>& memo,
    const vector<PathEntry>& path, Acc& acc) const;
  bool resolve_lattice(const OctBox& box, int32_t i_lat, int32_t& i_child,
    OctBox& child_box, array<int, 3>& idx) const;

  void credit(int32_t i_cell, int32_t i_univ, const vector<PathEntry>& path,
    double vol, Acc& acc) const;
  void charge_slack(int32_t i_univ, double vol, Acc& acc) const;

  //! Distribcell instance of a cell reached along `path`, mirroring
  //! cell_instance_at_level() in geometry.cpp so the two cannot drift.
  int instance_of(const Cell& c, const vector<PathEntry>& path) const;

  int register_pair(int slot, int32_t i_mat);
  int find_pair(int slot, int32_t i_mat) const;
  const Reach& reach(int32_t i_univ);

  DomainType type_;
  vector<int> domain_ids_;
  OctBox root_;
  vector<VolumeBracket> results_;
  int depth_reached_ {0};

  vector<double> lower_, slack_, mat_lower_, mat_slack_;
  vector<int32_t> pair_mat_; //!< material index of each (domain, material) pair
  vector<int> pair_slot_;    //!< domain slot of each pair
  std::unordered_map<int64_t, int> pair_index_;

  vector<Reach> reach_;
  vector<SurfaceProxy> proxies_;

  // Regions are stored in infix form and generate_postfix() re-runs
  // shunting-yard on every call, so the RPN is built once at construction.
  vector<vector<int32_t>> postfix_;
  vector<char> simple_;
  vector<char> is_csg_;

  //! Set when a box finds overlapping cells. Written from inside the parallel
  //! region, so it is a flag rather than a warning() call: reporting there
  //! could fire on millions of boxes from any thread.
  mutable std::atomic<int> overlap_universe_ {-1};

  std::unordered_map<int32_t, int> cell_slot_;
  std::unordered_map<int32_t, int> mat_slot_;
  std::unordered_map<int32_t, int> univ_slot_;
};

} // namespace openmc

#endif // OPENMC_VOLUME_OCTREE_H
