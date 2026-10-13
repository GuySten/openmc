#include "openmc/ifp.h"

#include <algorithm> // for fill, max, min

#include <fmt/core.h>

#include "openmc/bank.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/vector.h"

namespace openmc {

void ifp(const Particle& p, int64_t idx, const double* dg_weights)
{
  if (settings::ifp_delayed_group_on) {
    const auto& delayed_groups =
      simulation::ifp_source_delayed_group_bank[p.current_work()];
    simulation::ifp_fission_delayed_group_bank[idx] =
      _ifp(p.delayed_group(), delayed_groups);
  }
  if (settings::ifp_lifetime_on) {
    const auto& lifetimes =
      simulation::ifp_source_lifetime_bank[p.current_work()];
    simulation::ifp_fission_lifetime_bank[idx] = _ifp(p.lifetime(), lifetimes);
  }
  if (settings::ifp_dg_weight_on) {
    // Unlike the banks above, each list ends with the site's own birth so
    // that the probabilities computed when it was emitted are not lost. The
    // list of a neutron that was not born from fission is empty; it gets an
    // all-prompt entry, as the delayed group bank records group 0 for it.
    constexpr int stride = MAX_DELAYED_GROUPS;
    const auto& parent =
      simulation::ifp_source_dg_weight_bank[p.current_work()];
    auto& updated = simulation::ifp_fission_dg_weight_bank[idx];
    if (parent.empty()) {
      updated.assign(stride, 0.0);
    } else {
      updated = parent;
    }
    updated.insert(updated.end(), dg_weights, dg_weights + stride);
    size_t max_size = (settings::ifp_n_generation + 1) * stride;
    if (updated.size() > max_size) {
      updated.erase(
        updated.begin(), updated.begin() + (updated.size() - max_size));
    }
  }
}

void ifp_delayed_group_weights(int i_nuclide, const Reaction& rx, double E_in,
  double E_out, int delayed_group, double* weights)
{
  std::fill(weights, weights + MAX_DELAYED_GROUPS, 0.0);

  // Delayed emission probabilities, following sample_fission_neutron(): the
  // neutron is delayed with probability nu_d / nu_t, and the group is then
  // chosen from the cumulative yields, the last group taking any remainder
  const auto& nuc {data::nuclides[i_nuclide]};
  int n_group = std::min(nuc->n_precursor_, MAX_DELAYED_GROUPS);
  double nu_t = nuc->nu(E_in, Nuclide::EmissionMode::total);
  double nu_d = nuc->nu(E_in, Nuclide::EmissionMode::delayed);
  if (n_group == 0 || nu_d <= 0.0 || nu_t <= 0.0)
    return;
  double beta = nu_d / nu_t;

  double prob_group[MAX_DELAYED_GROUPS] = {0.};
  double prob_used = 0.0;
  for (int g = 1; g <= n_group; ++g) {
    double prob = (g < n_group) ? (*rx.products_[g].yield_)(E_in) / nu_d
                                : std::max(0.0, 1.0 - prob_used);
    prob_group[g - 1] = beta * prob;
    prob_used += prob;
  }

  // Mixture density of the outgoing energy and its delayed components
  auto density = [&](int i_product) {
    double f = rx.products_[i_product].energy_pdf(E_in, E_out);
    if (f < 0.0) {
      fatal_error(fmt::format("The outgoing energy density of fission product "
                              "{} of {} cannot be evaluated, which the "
                              "ifp-beta-numerator-cond and "
                              "ifp-beta-numerator-group scores require.",
        i_product, nuc->name_));
    }
    return f;
  };

  double total = (1.0 - beta) * density(0);
  for (int g = 1; g <= n_group; ++g) {
    weights[g - 1] = prob_group[g - 1] * density(g);
    total += weights[g - 1];
  }

  if (total > 0.0) {
    for (int g = 0; g < n_group; ++g)
      weights[g] /= total;
  } else {
    // The tabulated density vanishes where a neutron was nonetheless sampled
    // (e.g. exactly at a table edge); fall back to the analog indicator
    std::fill(weights, weights + MAX_DELAYED_GROUPS, 0.0);
    if (delayed_group > 0)
      weights[delayed_group - 1] = 1.0;
  }
}

void resize_simulation_ifp_banks()
{
  resize_ifp_data(simulation::ifp_source_delayed_group_bank,
    simulation::ifp_source_lifetime_bank, simulation::work_per_rank);
  resize_ifp_data(simulation::ifp_fission_delayed_group_bank,
    simulation::ifp_fission_lifetime_bank, 3 * simulation::work_per_rank);
  if (settings::ifp_dg_weight_on) {
    simulation::ifp_source_dg_weight_bank.resize(simulation::work_per_rank);
    simulation::ifp_fission_dg_weight_bank.resize(
      3 * simulation::work_per_rank);
  }
}

void copy_ifp_data_from_fission_banks(int64_t i_bank, int64_t i_temp,
  vector<vector<int>>& delayed_groups, vector<vector<double>>& lifetimes,
  vector<vector<double>>& dg_weights)
{
  if (settings::ifp_dg_weight_on) {
    dg_weights[i_temp] = simulation::ifp_fission_dg_weight_bank[i_bank];
  }
  if (settings::ifp_delayed_group_on) {
    delayed_groups[i_temp] = simulation::ifp_fission_delayed_group_bank[i_bank];
  }
  if (settings::ifp_lifetime_on) {
    lifetimes[i_temp] = simulation::ifp_fission_lifetime_bank[i_bank];
  }
}

#ifdef OPENMC_MPI
void broadcast_ifp_n_generation(int& n_generation,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes)
{
  if (mpi::rank == 0) {
    if (settings::ifp_delayed_group_on) {
      n_generation = static_cast<int>(delayed_groups[0].size());
    } else {
      n_generation = static_cast<int>(lifetimes[0].size());
    }
  }
  MPI_Bcast(&n_generation, 1, MPI_INT, 0, mpi::intracomm);
}

void copy_partial_ifp_data_to_source_banks(int64_t idx, int n, int64_t i_bank,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes,
  const vector<vector<double>>& dg_weights)
{
  if (settings::ifp_dg_weight_on) {
    std::copy(&dg_weights[idx], &dg_weights[idx + n],
      &simulation::ifp_source_dg_weight_bank[i_bank]);
  }
  if (settings::ifp_delayed_group_on) {
    std::copy(&delayed_groups[idx], &delayed_groups[idx + n],
      &simulation::ifp_source_delayed_group_bank[i_bank]);
  }
  if (settings::ifp_lifetime_on) {
    std::copy(&lifetimes[idx], &lifetimes[idx + n],
      &simulation::ifp_source_lifetime_bank[i_bank]);
  }
}
#endif

void copy_complete_ifp_data_to_source_banks(
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes,
  const vector<vector<double>>& dg_weights)
{
  if (settings::ifp_dg_weight_on) {
    std::copy(dg_weights.data(), dg_weights.data() + settings::n_particles,
      simulation::ifp_source_dg_weight_bank.begin());
  }
  if (settings::ifp_delayed_group_on) {
    std::copy(delayed_groups.data(),
      delayed_groups.data() + settings::n_particles,
      simulation::ifp_source_delayed_group_bank.begin());
  }
  if (settings::ifp_lifetime_on) {
    std::copy(lifetimes.data(), lifetimes.data() + settings::n_particles,
      simulation::ifp_source_lifetime_bank.begin());
  }
}

void allocate_temporary_vector_ifp(vector<vector<int>>& delayed_groups,
  vector<vector<double>>& lifetimes, vector<vector<double>>& dg_weights)
{
  if (settings::ifp_dg_weight_on) {
    dg_weights.resize(simulation::fission_bank.size());
  }
  if (settings::ifp_delayed_group_on) {
    delayed_groups.resize(simulation::fission_bank.size());
  }
  if (settings::ifp_lifetime_on) {
    lifetimes.resize(simulation::fission_bank.size());
  }
}

void copy_ifp_data_to_fission_banks(const vector<int>* const delayed_groups_ptr,
  const vector<double>* lifetimes_ptr, const vector<double>* dg_weights_ptr)
{
  if (settings::ifp_dg_weight_on) {
    std::copy(dg_weights_ptr, dg_weights_ptr + simulation::fission_bank.size(),
      simulation::ifp_fission_dg_weight_bank.data());
  }
  if (settings::ifp_delayed_group_on) {
    std::copy(delayed_groups_ptr,
      delayed_groups_ptr + simulation::fission_bank.size(),
      simulation::ifp_fission_delayed_group_bank.data());
  }
  if (settings::ifp_lifetime_on) {
    std::copy(lifetimes_ptr, lifetimes_ptr + simulation::fission_bank.size(),
      simulation::ifp_fission_lifetime_bank.data());
  }
}

} // namespace openmc
