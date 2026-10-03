#include "openmc/track_output.h"

#include "openmc/constants.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/position.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/vector.h"

#include "openmc/tensor.h"
#include <fmt/core.h>
#include <hdf5.h>

#include <algorithm> // for lower_bound, sort
#include <cstddef>   // for size_t
#include <string>
#include <utility> // for pair

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

hid_t track_file;     //! HDF5 identifier for track file
hid_t track_dtype;    //! HDF5 identifier for track datatype
int n_tracks_written; //! Number of tracks written

namespace {

//! (root index, particle ID) of each source particle whose track is being
//! written, recorded on this rank while the source particles are transported
//! with the shared secondary bank
vector<std::pair<int64_t, int64_t>> pending_track_roots;

//! Pairs recorded on all ranks, sorted by root index
vector<std::pair<int64_t, int64_t>> track_roots;

} // namespace

//==============================================================================
// Non-member functions
//==============================================================================

void add_particle_track(Particle& p)
{
  auto& track = p.tracks().emplace_back();
  track.particle = p.type();
}

void write_particle_track(Particle& p)
{
  p.tracks().back().states.push_back(p.get_track_state());
}

void open_track_file()
{
  // Open file and write filetype/version -- when MPI is enabled and there is
  // more than one rank, each rank writes its own file
#ifdef OPENMC_MPI
  std::string filename;
  if (mpi::n_procs > 1) {
    filename = fmt::format("{}tracks_p{}.h5", settings::path_output, mpi::rank);
  } else {
    filename = fmt::format("{}tracks.h5", settings::path_output);
  }
#else
  std::string filename = fmt::format("{}tracks.h5", settings::path_output);
#endif
  track_file = file_open(filename, 'w');
  write_attribute(track_file, "filetype", "track");
  write_attribute(track_file, "version", VERSION_TRACK);

  // Create compound type for Position
  hid_t postype = H5Tcreate(H5T_COMPOUND, sizeof(struct Position));
  H5Tinsert(postype, "x", HOFFSET(Position, x), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "y", HOFFSET(Position, y), H5T_NATIVE_DOUBLE);
  H5Tinsert(postype, "z", HOFFSET(Position, z), H5T_NATIVE_DOUBLE);

  // Create compound type for TrackState
  track_dtype = H5Tcreate(H5T_COMPOUND, sizeof(struct TrackState));
  H5Tinsert(track_dtype, "r", HOFFSET(TrackState, r), postype);
  H5Tinsert(track_dtype, "u", HOFFSET(TrackState, u), postype);
  H5Tinsert(track_dtype, "E", HOFFSET(TrackState, E), H5T_NATIVE_DOUBLE);
  H5Tinsert(track_dtype, "time", HOFFSET(TrackState, time), H5T_NATIVE_DOUBLE);
  H5Tinsert(track_dtype, "wgt", HOFFSET(TrackState, wgt), H5T_NATIVE_DOUBLE);
  H5Tinsert(
    track_dtype, "cell_id", HOFFSET(TrackState, cell_id), H5T_NATIVE_INT);
  H5Tinsert(track_dtype, "cell_instance", HOFFSET(TrackState, cell_instance),
    H5T_NATIVE_INT);
  H5Tinsert(track_dtype, "material_id", HOFFSET(TrackState, material_id),
    H5T_NATIVE_INT);
  H5Tclose(postype);
}

void close_track_file()
{
  H5Tclose(track_dtype);
  file_close(track_file);

  // Reset number of tracks written
  n_tracks_written = 0;
}

bool check_track_criteria(const Particle& p)
{
  if (settings::write_all_tracks) {
    // Increment number of tracks written and get previous value
    int n;
#pragma omp atomic capture
    n = n_tracks_written++;

    // Indicate that track should be written for this particle
    return n < settings::max_tracks;
  }

  // Check for match from explicit track identifiers
  if (settings::track_identifiers.size() > 0) {
    for (const auto& t : settings::track_identifiers) {
      if (simulation::current_batch == t[0] &&
          simulation::current_gen == t[1] && p.id() == t[2]) {
        return true;
      }
    }
  }
  return false;
}

void finalize_particle_track(Particle& p)
{
  // Determine number of coordinates for each particle
  vector<int> offsets;
  vector<int> particles;
  vector<TrackState> tracks;
  int offset = 0;
  for (auto& track_i : p.tracks()) {
    offsets.push_back(offset);
    particles.push_back(track_i.particle.pdg_number());
    offset += track_i.states.size();
    tracks.insert(tracks.end(), track_i.states.begin(), track_i.states.end());
  }
  offsets.push_back(offset);

#pragma omp critical(FinalizeParticleTrack)
  {
    // Create name for dataset
    std::string dset_name = fmt::format("track_{}_{}_{}",
      simulation::current_batch, simulation::current_gen, p.id());

    // Write array of TrackState to file
    hsize_t dims[] {static_cast<hsize_t>(tracks.size())};
    hid_t dspace = H5Screate_simple(1, dims, nullptr);
    hid_t dset = H5Dcreate(track_file, dset_name.c_str(), track_dtype, dspace,
      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, track_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tracks.data());

    // Write attributes
    write_attribute(dset, "n_particles", p.tracks().size());
    write_attribute(dset, "offsets", offsets);
    write_attribute(dset, "particles", particles);

    // With the shared secondary bank, a secondary is written to its own
    // dataset; link it to the dataset of the source particle of its history
    if (settings::use_shared_secondary_bank) {
      int64_t root_id = track_root_id(p.root_index());
      if (root_id >= 0 && root_id != p.id()) {
        write_attribute(dset, "root_id", root_id);
      }
    }

    // Free resources
    H5Dclose(dset);
    H5Sclose(dspace);
  }

  // Clear particle tracks
  p.tracks().clear();
}

void reset_track_roots()
{
  pending_track_roots.clear();
  track_roots.clear();
}

void record_track_root(const Particle& p)
{
#pragma omp critical(RecordTrackRoot)
  pending_track_roots.emplace_back(p.root_index(), p.id());
}

void synchronize_track_roots()
{
  track_roots = pending_track_roots;
  pending_track_roots.clear();

#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    // Flatten pairs and gather them from all ranks
    vector<int64_t> send;
    for (const auto& [root, id] : track_roots) {
      send.push_back(root);
      send.push_back(id);
    }
    int n_send = send.size();
    vector<int> counts(mpi::n_procs);
    MPI_Allgather(
      &n_send, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi::intracomm);
    vector<int> displs(mpi::n_procs, 0);
    for (int i = 1; i < mpi::n_procs; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
    vector<int64_t> recv(displs.back() + counts.back());
    MPI_Allgatherv(send.data(), n_send, MPI_INT64_T, recv.data(), counts.data(),
      displs.data(), MPI_INT64_T, mpi::intracomm);

    track_roots.clear();
    for (std::size_t i = 0; i < recv.size(); i += 2) {
      track_roots.emplace_back(recv[i], recv[i + 1]);
    }
  }
#endif

  std::sort(track_roots.begin(), track_roots.end());
}

int64_t track_root_id(int64_t root_index)
{
  auto it = std::lower_bound(track_roots.begin(), track_roots.end(), root_index,
    [](const auto& a, int64_t root) { return a.first < root; });
  if (it != track_roots.end() && it->first == root_index) {
    return it->second;
  }
  return -1;
}

} // namespace openmc
