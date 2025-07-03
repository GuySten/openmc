#include "openmc/ifp.h"

#include "openmc/bank.h"
#include "openmc/message_passing.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/vector.h"

namespace openmc {

bool is_external_source()
{
  return settings::ifp_settings.test(IFPParameter::ExternalSource);
}

bool is_beta_effective()
{
  return settings::ifp_settings.test(IFPParameter::BetaEffective);
}

bool is_generation_time()
{
  return settings::ifp_settings.test(IFPParameter::GenerationTime);
}

void ifp(const Particle& p, const SourceSite& site, int64_t idx)
{
  if (is_beta_effective()) {
    const auto& delayed_groups =
      simulation::ifp_source_delayed_group_bank[p.current_work() - 1];
    simulation::ifp_fission_delayed_group_bank[idx] =
      _ifp(site.delayed_group, delayed_groups);
  }
  if (is_generation_time()) {
    const auto& lifetimes =
      simulation::ifp_source_lifetime_bank[p.current_work() - 1];
    simulation::ifp_fission_lifetime_bank[idx] = _ifp(p.lifetime(), lifetimes);
  }
  if (is_external_source()) {
    const auto& ext_srcs =
      simulation::ifp_source_ext_src_bank[p.current_work() - 1];
    simulation::ifp_fission_ext_src_bank[idx] = _ifp(p.wgt(), ext_srcs);
  }
}

void resize_simulation_ifp_banks()
{
  resize_ifp_data(simulation::ifp_source_delayed_group_bank,
    simulation::ifp_source_lifetime_bank, simulation::ifp_source_ext_src_bank,
    simulation::work_per_rank);
  resize_ifp_data(simulation::ifp_fission_delayed_group_bank,
    simulation::ifp_fission_lifetime_bank, simulation::ifp_fission_ext_src_bank,
    3 * simulation::work_per_rank);
}

void copy_ifp_data_from_fission_banks(int i_bank, vector<int>& delayed_groups,
  vector<double>& lifetimes, vector<double>& ext_srcs)
{
  if (is_beta_effective()) {
    delayed_groups = simulation::ifp_fission_delayed_group_bank[i_bank];
  }
  if (is_generation_time()) {
    lifetimes = simulation::ifp_fission_lifetime_bank[i_bank];
  }
  if (is_external_source()) {
    ext_srcs = simulation::ifp_fission_ext_src_bank[i_bank];
  }
}

#ifdef OPENMC_MPI

void broadcast_ifp_n_generation(int& n_generation,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes,
  const vector<vector<double>>& ext_srcs)
{
  if (mpi::rank == 0) {
    if (is_beta_effective()) {
      n_generation = static_cast<int>(delayed_groups[0].size());
    } else {
      n_generation = static_cast<int>(lifetimes[0].size());
    }
  }
  MPI_Bcast(&n_generation, 1, MPI_INT, 0, mpi::intracomm);
}

void send_ifp_info(int64_t idx, int64_t n, int n_generation, int neighbor,
  vector<MPI_Request>& requests, const vector<vector<int>>& delayed_groups,
  vector<int>& send_delayed_groups, const vector<vector<double>>& lifetimes,
  vector<double>& send_lifetimes, const vector<vector<double>>& ext_srcs,
  vector<double>& send_ext_srcs)
{
  // Copy data in send buffers
  for (int i = idx; i < idx + n; i++) {
    if (is_beta_effective()) {
      std::copy(delayed_groups[i].begin(), delayed_groups[i].end(),
        send_delayed_groups.begin() + i * n_generation);
    }
    if (is_generation_time()) {
      std::copy(lifetimes[i].begin(), lifetimes[i].end(),
        send_lifetimes.begin() + i * n_generation);
    }
    if (is_external_source()) {
      std::copy(ext_srcs[i].begin(), ext_srcs[i].end(),
        send_ext_srcs.begin() + i * n_generation);
    }
  }
  // Send delayed groups
  if (is_beta_effective()) {
    requests.emplace_back();
    MPI_Isend(&send_delayed_groups[n_generation * idx],
      n_generation * static_cast<int>(n), MPI_INT, neighbor, mpi::rank,
      mpi::intracomm, &requests.back());
  }
  // Send lifetimes
  if (is_generation_time()) {
    requests.emplace_back();
    MPI_Isend(&send_lifetimes[n_generation * idx],
      n_generation * static_cast<int>(n), MPI_DOUBLE, neighbor, mpi::rank,
      mpi::intracomm, &requests.back());
  }
  // Send external sources
  if (is_external_source()) {
    requests.emplace_back();
    MPI_Isend(&send_ext_srcs[n_generation * idx],
      n_generation * static_cast<int>(n), MPI_DOUBLE, neighbor, mpi::rank,
      mpi::intracomm, &requests.back());
  }
}

void receive_ifp_data(int64_t idx, int64_t n, int n_generation, int neighbor,
  vector<MPI_Request>& requests, vector<int>& delayed_groups,
  vector<double>& lifetimes, vector<double>& ext_srcs,
  vector<DeserializationInfo>& deserialization)
{
  // Receive delayed groups
  if (is_beta_effective()) {
    requests.emplace_back();
    MPI_Irecv(&delayed_groups[n_generation * idx],
      n_generation * static_cast<int>(n), MPI_INT, neighbor, neighbor,
      mpi::intracomm, &requests.back());
  }
  // Receive lifetimes
  if (is_generation_time()) {
    requests.emplace_back();
    MPI_Irecv(&lifetimes[n_generation * idx],
      n_generation * static_cast<int>(n), MPI_DOUBLE, neighbor, neighbor,
      mpi::intracomm, &requests.back());
  }
  // Receive external sources
  if (is_external_source()) {
    requests.emplace_back();
    MPI_Irecv(&ext_srcs[n_generation * idx], n_generation * static_cast<int>(n),
      MPI_DOUBLE, neighbor, neighbor, mpi::intracomm, &requests.back());
  }

  // Deserialization info to reconstruct data later
  DeserializationInfo info = {idx, n};
  deserialization.push_back(info);
}

void copy_partial_ifp_data_to_source_banks(int64_t idx, int n, int64_t i_bank,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes,
  const vector<vector<double>>& ext_srcs)
{
  if (is_beta_effective()) {
    std::copy(&delayed_groups[idx], &delayed_groups[idx + n],
      &simulation::ifp_source_delayed_group_bank[i_bank]);
  }
  if (is_generation_time()) {
    std::copy(&lifetimes[idx], &lifetimes[idx + n],
      &simulation::ifp_source_lifetime_bank[i_bank]);
  }
  if (is_external_source()) {
    std::copy(&ext_srcs[idx], &ext_srcs[idx + n],
      &simulation::ifp_source_ext_src_bank[i_bank]);
  }
}

void deserialize_ifp_info(int n_generation,
  const vector<DeserializationInfo>& deserialization,
  const vector<int>& delayed_groups, const vector<double>& lifetimes,
  const vector<double>& ext_srcs)
{
  for (auto info : deserialization) {
    int64_t index_local = info.index_local;
    int64_t n = info.n;

    for (int i = index_local; i < index_local + n; i++) {
      if (is_beta_effective()) {
        vector<int> delayed_groups_received(
          delayed_groups.begin() + n_generation * i,
          delayed_groups.begin() + n_generation * (i + 1));
        simulation::ifp_source_delayed_group_bank[i] = delayed_groups_received;
      }
      if (is_generation_time()) {
        vector<double> lifetimes_received(lifetimes.begin() + n_generation * i,
          lifetimes.begin() + n_generation * (i + 1));
        simulation::ifp_source_lifetime_bank[i] = lifetimes_received;
      }
      if (is_external_source()) {
        vector<double> ext_src_received(ext_srcs.begin() + n_generation * i,
          ext_srcs.begin() + n_generation * (i + 1));
        simulation::ifp_source_ext_src_bank[i] = ext_src_received;
      }
    }
  }
}

#endif

void copy_complete_ifp_data_to_source_banks(
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes,
  const vector<vector<double>>& ext_srcs)
{
  if (is_beta_effective()) {
    std::copy(delayed_groups.data(),
      delayed_groups.data() + settings::n_particles,
      simulation::ifp_source_delayed_group_bank.begin());
  }
  if (is_generation_time()) {
    std::copy(lifetimes.data(), lifetimes.data() + settings::n_particles,
      simulation::ifp_source_lifetime_bank.begin());
  }
  if (is_external_source()) {
    std::copy(ext_srcs.data(), ext_srcs.data() + settings::n_particles,
      simulation::ifp_source_ext_src_bank.begin());
  }
}

void allocate_temporary_vector_ifp(vector<vector<int>>& delayed_groups,
  vector<vector<double>>& lifetimes, vector<vector<double>>& ext_srcs)
{
  if (is_beta_effective()) {
    delayed_groups.resize(simulation::fission_bank.size());
  }
  if (is_generation_time()) {
    lifetimes.resize(simulation::fission_bank.size());
  }
  if (is_external_source()) {
    ext_srcs.resize(simulation::fission_bank.size());
  }
}

void copy_ifp_data_to_fission_banks(const vector<int>* const delayed_groups_ptr,
  const vector<double>* lifetimes_ptr, const vector<double>* ext_srcs_ptr)
{
  if (is_beta_effective()) {
    std::copy(delayed_groups_ptr,
      delayed_groups_ptr + simulation::fission_bank.size(),
      simulation::ifp_fission_delayed_group_bank.data());
  }
  if (is_generation_time()) {
    std::copy(lifetimes_ptr, lifetimes_ptr + simulation::fission_bank.size(),
      simulation::ifp_fission_lifetime_bank.data());
  }
  if (is_external_source()) {
    std::copy(ext_srcs_ptr, ext_srcs_ptr + simulation::fission_bank.size(),
      simulation::ifp_fission_ext_src_bank.data());
  }
}

} // namespace openmc
