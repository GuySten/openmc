#ifndef OPENMC_BANK_H
#define OPENMC_BANK_H

#include <cstdint>

#include "openmc/particle.h"
#include "openmc/position.h"
#include "openmc/shared_array.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace simulation {

extern vector<SourceSite> source_bank;

extern SharedArray<SourceSite> surf_source_bank;

// Surface source sites carry two levels of structure so that a calculation
// reading the file can reconstruct which sites belong together.
//
// A *group* is the set of sites banked by one source history. Its sites must be
// emitted as a single history by any calculation reading the file, both because
// they are correlated with one another and because per-history scores such as
// pulse-height tallies are otherwise split into several smaller scores. Threads
// append to the bank as they go, so a history's sites are made contiguous by
// sorting each generation's range on parent_id in
// surf_source_close_generation().
//
// A *batch* is the set of groups banked during one batch. Batches are
// statistically independent of one another and carry the number of source
// particles simulated, including those that produced no site at all, which is
// the denominator any normalization or variance estimate needs. This mirrors
// the MCNP surface source convention, where each track records the history
// number that produced it and the header records NP1, the number of histories
// run, separately from NRSS, the number of tracks stored.
//
// Sites are written to the file rank-major, so the groups of one batch are
// contiguous within a rank but not across ranks. The file therefore stores one
// entry per rank and batch, along with the rank count needed to interpret
// them; a reader merges same-numbered batches across ranks so that the batch
// count, the source particle count of each batch, and the unit of statistical
// independence are all the same whether the file was written on one rank or
// many.

//! Index into surf_source_bank at which each group begins. Local to this rank.
extern vector<int64_t> ssw_group_offsets;

//! Index into ssw_group_offsets just past the last group of each batch
extern vector<int64_t> ssw_batch_group_end;

//! Number of source particles simulated by this rank for each batch
extern vector<int64_t> ssw_batch_n_particles;

//! Whether each batch is complete, i.e. no site belonging to it was discarded
//! because the surface source bank filled up partway through
extern vector<int> ssw_batch_complete;

//! Index into surf_source_bank at which the current generation began
extern int64_t ssw_gen_start;

//! Whether any site that passed the surface and cell filters was discarded
//! from the batch currently being accumulated because the bank was full
extern int ssw_batch_truncated;

extern SharedArray<CollisionTrackSite> collision_track_bank;

extern SharedArray<SourceSite> fission_bank;

extern vector<vector<int>> ifp_source_delayed_group_bank;

extern vector<vector<double>> ifp_source_lifetime_bank;

extern vector<vector<int>> ifp_fission_delayed_group_bank;

extern vector<vector<double>> ifp_fission_lifetime_bank;

extern vector<int64_t> progeny_per_particle;

extern SharedArray<SourceSite> shared_secondary_bank_read;
extern SharedArray<SourceSite> shared_secondary_bank_write;

} // namespace simulation

//==============================================================================
// Non-member functions
//==============================================================================

void sort_bank(SharedArray<SourceSite>& bank, bool is_fission_bank);

//==============================================================================
// Surface source grouping
//==============================================================================

//! Group and batch boundaries of a surface source file, indexed against the
//! file rather than against any one rank's bank
struct SurfaceSourceGroups {
  //! Site index at which each group begins, with a trailing total
  vector<int64_t> group_offsets;
  //! Group index at which each rank-batch segment begins, with a trailing
  //! total. Segment s = rank * n_batches + batch.
  vector<int64_t> batch_offsets;
  //! Source particles simulated for each rank-batch segment
  vector<int64_t> batch_n_particles;
  //! Whether each rank-batch segment lost no site to a full bank
  vector<int> batch_complete;
  //! Number of ranks the segments are spread over, needed to merge them
  int n_ranks {1};

  //! Total number of source particles represented by the file
  int64_t n_source_particles() const;
};

//! Sort the sites banked during the generation that just finished so that the
//! sites of each history are contiguous, and record where each history begins
void surf_source_close_generation();

//! Record the extent of the batch that just finished and the number of source
//! particles that produced it
void surf_source_close_batch();

//! Discard all group bookkeeping, for use after a surface source file is
//! written and the bank is cleared
void surf_source_reset_groups();

//! Collect the group and batch boundaries of all ranks onto the master
//!
//! \param bank_index Cumulative site counts per rank, as used when writing the
//!   source bank itself
//! \return Fully populated on the master rank, empty elsewhere
SurfaceSourceGroups gather_surface_source_groups(
  const vector<int64_t>& bank_index);

void free_memory_bank();

void init_fission_bank(int64_t max);

int64_t synchronize_global_secondary_bank(
  SharedArray<SourceSite>& shared_secondary_bank);

} // namespace openmc

#endif // OPENMC_BANK_H
