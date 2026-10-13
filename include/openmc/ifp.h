#ifndef OPENMC_IFP_H
#define OPENMC_IFP_H

#include "openmc/message_passing.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/reaction.h"
#include "openmc/settings.h"

#include <algorithm> // for copy

namespace openmc {

//! Resize IFP vectors
//!
//! \param[in,out] delayed_groups List of delayed group numbers
//! \param[in,out] lifetimes List of lifetimes
//! \param[in] n  Dimension to resize vectors
template<typename T, typename U>
void resize_ifp_data(vector<T>& delayed_groups, vector<U>& lifetimes, int64_t n)
{
  if (settings::ifp_delayed_group_on) {
    delayed_groups.resize(n);
  }
  if (settings::ifp_lifetime_on) {
    lifetimes.resize(n);
  }
}

//! Update a list of values by adding a new value if the size
//! of the list can accomodate the new value or by shifting all
//! values to the left (removing the first value of the list
//! and adding the new value at the end of the list).
//!
//! \param[in] value Value to add to the list
//! \param[in] data Initial version of the list
//! \return Updated list
template<typename T>
vector<T> _ifp(const T& value, const vector<T>& data)
{
  vector<T> updated;
  size_t source_idx = data.size();

  if (source_idx < settings::ifp_n_generation) {
    updated.resize(source_idx + 1);
    for (size_t i = 0; i < source_idx; i++) {
      updated[i] = data[i];
    }
    updated[source_idx] = value;
  } else if (source_idx == settings::ifp_n_generation) {
    updated.resize(source_idx);
    for (size_t i = 0; i < source_idx - 1; i++) {
      updated[i] = data[i + 1];
    }
    updated[source_idx - 1] = value;
  }
  return updated;
}

//! \brief Iterated Fission Probability (IFP) method.
//!
//! Add the IFP information in the IFP banks using the same index
//! as the one used to append the fission site to the fission bank.
//! The information stored are the delayed group number and lifetime
//! of the neutron that created the fission event.
//! Multithreading protection is guaranteed by the index returned by the
//! thread_safe_append call in physics.cpp.
//!
//! \param[in] p Particle
//! \param[in] idx Bank index from the thread_safe_append call in physics.cpp
//! \param[in] dg_weights Delayed group probabilities of the new fission site
//!   (MAX_DELAYED_GROUPS values), only read when settings::ifp_dg_weight_on
void ifp(const Particle& p, int64_t idx, const double* dg_weights);

//! Compute the probability that a fission neutron was emitted in each delayed
//! group given the fission event and its sampled outgoing energy.
//!
//! The neutron is sampled from the mixture
//!   f(E_out) = (1 - beta) chi_p(E_out) + sum_g beta_g chi_g(E_out),
//! so the probability of group g given E_out is beta_g chi_g(E_out) / f(E_out).
//! Scoring these probabilities in place of the 0/1 delayed indicator gives the
//! same expected IFP beta numerator, because the progeny of the neutron depend
//! on how it was emitted only through its outgoing energy.
//!
//! \param[in] i_nuclide Index of the fissioning nuclide
//! \param[in] rx Fission reaction that produced the neutron
//! \param[in] E_in Energy of the neutron causing fission in [eV]
//! \param[in] E_out Sampled energy of the fission neutron in [eV]
//! \param[in] delayed_group Sampled delayed group (0 for prompt), used only
//!   when the mixture density vanishes at E_out
//! \param[out] weights Probabilities for groups 1..MAX_DELAYED_GROUPS
void ifp_delayed_group_weights(int i_nuclide, const Reaction& rx, double E_in,
  double E_out, int delayed_group, double* weights);

//! Resize the IFP banks used in the simulation
void resize_simulation_ifp_banks();

//! Retrieve IFP data from the IFP fission banks.
//!
//! \param[in] i_bank Index in the fission banks
//! \param[in] i_temp Index in the temporary bank vectors
//! \param[in,out] delayed_groups Delayed group numbers
//! \param[in,out] lifetimes Lifetimes lists
//! \param[in,out] dg_weights Delayed group probability lists
void copy_ifp_data_from_fission_banks(int64_t i_bank, int64_t i_temp,
  vector<vector<int>>& delayed_groups, vector<vector<double>>& lifetimes,
  vector<vector<double>>& dg_weights);

#ifdef OPENMC_MPI

//! Deserialization information for transfer of IFP data using MPI
struct DeserializationInfo {
  int64_t index_local; //!< local index
  int64_t n;           //!< number of sites sent
};

//! Broadcast the number of generation determined by the size of the first
//! element on the first processor.
//!
//! \param[in] n_generation Number of generations
//! \param[in] delayed_groups List of delayed group numbers lists
//! \param[in] lifetimes List of lifetimes lists
void broadcast_ifp_n_generation(int& n_generation,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes);

//! Send IFP data using MPI.
//!
//! \param[in] idx Index of the first site
//! \param[in] n Number of sites to send
//! \param[in] n_generation Number of generations
//! \param[in] neighbor Index of the neighboring processor
//! \param[in] requests MPI requests
//! \param[in] data List of data lists
//! \param[out] send_data data buffer
template<typename T>
void send_ifp_info(int64_t idx, int64_t n, int n_generation, int neighbor,
  vector<MPI_Request>& requests, const vector<vector<T>>& data,
  vector<T>& send_data)
{
  // Copy data in buffer
  for (int i = idx; i < idx + n; i++) {
    std::copy(
      data[i].begin(), data[i].end(), send_data.begin() + i * n_generation);
  }

  // Send data
  requests.emplace_back();
  MPI_Datatype datatype = mpi::MPITypeMap<T>::mpi_type;
  MPI_Isend(&send_data[n_generation * idx], n_generation * static_cast<int>(n),
    datatype, neighbor, mpi::rank, mpi::intracomm, &requests.back());
}

//! Receive IFP data using MPI.
//!
//! \param[in] idx Index of the first site
//! \param[in] n Number of sites to receive
//! \param[in] n_generation Number of generations
//! \param[in] neighbor Index of the neighboring processor
//! \param[in] requests MPI requests
//! \param[in] data data buffer
//! \param[out] deserialization Information to deserialize the received data
template<typename T>
void receive_ifp_data(int64_t idx, int64_t n, int n_generation, int neighbor,
  vector<MPI_Request>& requests, vector<T>& data,
  vector<DeserializationInfo>& deserialization)
{
  requests.emplace_back();
  MPI_Datatype datatype = mpi::MPITypeMap<T>::mpi_type;
  MPI_Irecv(&data[n_generation * idx], n_generation * static_cast<int>(n),
    datatype, neighbor, neighbor, mpi::intracomm, &requests.back());

  // Deserialization info to reconstruct data later
  DeserializationInfo info = {idx, n};
  deserialization.push_back(info);
}

//! Copy partial IFP data from local lists to source banks.
//!
//! \param[in] idx Index of the first site
//! \param[in] n Number of sites to copy
//! \param[in] i_bank Index in the IFP source banks
//! \param[in] delayed_groups List of delayed group numbers lists
//! \param[in] lifetimes List of lifetimes lists
//! \param[in] dg_weights List of delayed group probability lists
void copy_partial_ifp_data_to_source_banks(int64_t idx, int n, int64_t i_bank,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes,
  const vector<vector<double>>& dg_weights);

//! Deserialize IFP information received using MPI and store it in
//! the IFP source banks.
//!
//! \param[in] n_generation Number of generations
//! \param[in] data data to deserialize
//! \param[in] bank bank to store data
//! \param[out] deserialization Information to deserialize the received data
template<typename T>
void deserialize_ifp_info(int n_generation, const vector<T>& data,
  vector<vector<T>>& bank, const vector<DeserializationInfo>& deserialization)
{
  for (auto info : deserialization) {
    int64_t index_local = info.index_local;
    int64_t n = info.n;

    for (int i = index_local; i < index_local + n; i++) {
      vector<T> data_received(
        data.begin() + n_generation * i, data.begin() + n_generation * (i + 1));
      bank[i] = data_received;
    }
  }
}

#endif

//! Copy IFP temporary vectors to source banks.
//!
//! \param[in] delayed_groups List of delayed group numbers lists
//! \param[in] lifetimes List of lifetimes lists
//! \param[in] dg_weights List of delayed group probability lists
void copy_complete_ifp_data_to_source_banks(
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes,
  const vector<vector<double>>& dg_weights);

//! Allocate temporary vectors for IFP data.
//!
//! \param[in,out] delayed_groups List of delayed group numbers lists
//! \param[in,out] lifetimes List of delayed group numbers lists
//! \param[in,out] dg_weights List of delayed group probability lists
void allocate_temporary_vector_ifp(vector<vector<int>>& delayed_groups,
  vector<vector<double>>& lifetimes, vector<vector<double>>& dg_weights);

//! Copy local IFP data to IFP fission banks.
//!
//! \param[in] delayed_groups_ptr Pointer to delayed group numbers
//! \param[in] lifetimes_ptr Pointer to lifetimes
//! \param[in] dg_weights_ptr Pointer to delayed group probabilities
void copy_ifp_data_to_fission_banks(const vector<int>* delayed_groups_ptr,
  const vector<double>* lifetimes_ptr, const vector<double>* dg_weights_ptr);

} // namespace openmc

#endif // OPENMC_IFP_H
