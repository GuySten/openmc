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

namespace simulation {

IFPStream<int> ifp_delayed_group;
IFPStream<double> ifp_lifetime;
IFPStream<double> ifp_dg_weight;

} // namespace simulation

void ifp(const Particle& p, int64_t idx, const double* dg_weights)
{
  simulation::ifp_delayed_group.store(p.current_work(), idx, p.delayed_group());
  simulation::ifp_lifetime.store(p.current_work(), idx, p.lifetime());

  auto& stream = simulation::ifp_dg_weight;
  if (stream.enabled()) {
    // Each list ends with the site's own birth so that the probabilities
    // computed when it was emitted are not lost. The list of a neutron that
    // was not born from fission is empty; it gets an all-prompt entry, as the
    // delayed group stream records group 0 for it.
    constexpr int stride = MAX_DELAYED_GROUPS;
    const auto& parent = stream.source_bank()[p.current_work()];
    auto& updated = stream.fission_bank()[idx];
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
  int64_t n = simulation::work_per_rank;
  simulation::ifp_delayed_group.resize_banks(n, 3 * n);
  simulation::ifp_lifetime.resize_banks(n, 3 * n);
  simulation::ifp_dg_weight.resize_banks(n, 3 * n);
}

void reset_ifp_streams()
{
  simulation::ifp_delayed_group.reset();
  simulation::ifp_lifetime.reset();
  simulation::ifp_dg_weight.reset();
}

#ifdef OPENMC_MPI
void broadcast_ifp_n_generation(int& n_generation,
  const vector<vector<int>>& delayed_groups,
  const vector<vector<double>>& lifetimes)
{
  if (mpi::rank == 0) {
    n_generation = simulation::ifp_delayed_group.enabled()
                     ? static_cast<int>(delayed_groups[0].size())
                     : static_cast<int>(lifetimes[0].size());
  }
  MPI_Bcast(&n_generation, 1, MPI_INT, 0, mpi::intracomm);
}
#endif

} // namespace openmc
