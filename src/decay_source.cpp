//! \file decay_source.cpp
//! \brief Steady-state sources of decay photons, electrons, and positrons

#include "openmc/decay_source.h"

#include <algorithm> // for sort, any_of
#include <cmath>     // for exp, expm1
#include <map>
#include <utility> // for pair

#include <fmt/core.h>

#include "openmc/chain.h"
#include "openmc/distribution_multi.h" // for isotropic_direction
#include "openmc/endf.h"               // for is_fission
#include "openmc/error.h"
#include "openmc/nuclide.h"
#include "openmc/random_lcg.h"
#include "openmc/reaction.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace settings {

bool decay_sources_on {false};
DecaySourceSettings decay_source_settings;

} // namespace settings

namespace data {

vector<NuclideDecaySource> decay_sources;

} // namespace data

//==============================================================================
// Transmutation trajectory analysis
//==============================================================================

namespace {

//! Particle types that can be emitted, in the order used for arrays
constexpr array<ParticleType, N_DECAY_PARTICLES> DECAY_PARTICLES {
  ParticleType::photon(), ParticleType::electron(), ParticleType::positron()};

//! Minimum relative separation between decay constants on a trajectory. The
//! Bateman solution is singular for equal decay constants, so a nuclide whose
//! decay constant is closer than this to that of an ancestor is given a
//! slightly shifted decay constant. The shift introduces a relative error of
//! the order of the separation while round-off in the Bateman coefficients
//! grows as the inverse of the separation, so the square root of the machine
//! epsilon balances the two.
constexpr double LAMBDA_SEPARATION {1.0e-8};

//! A trajectory is not followed past a nuclide through which less than this
//! fraction of the produced atoms has passed by the end of the history
constexpr double TTA_CUTOFF {1.0e-12};

//! Maximum number of nuclides on a trajectory
constexpr int TTA_MAX_DEPTH {100};

//! Irradiation history in the form used by the Bateman solution
struct History {
  vector<double> power; //!< Power relative to the final power
  vector<double> start; //!< Time from start of each step to end of history
  vector<double> end;   //!< Time from end of each step to end of history
  double produced;      //!< Atoms produced per unit final production rate
};

History make_history(const DecaySourceSettings& s)
{
  History h;
  double total_time = 0.0;
  for (double dt : s.timesteps) {
    total_time += dt;
  }

  double final_power = s.power.back();
  double t = 0.0;
  h.produced = 0.0;
  for (size_t i = 0; i < s.timesteps.size(); ++i) {
    double p = s.power[i] / final_power;
    h.power.push_back(p);
    h.start.push_back(total_time - t);
    t += s.timesteps[i];
    h.end.push_back(total_time - t);
    h.produced += p * s.timesteps[i];
  }
  return h;
}

//! Integral of exp(-lambda*t) for t from a to b
double exp_integral(double lambda, double a, double b)
{
  if (lambda == 0.0)
    return b - a;
  return std::exp(-lambda * a) * -std::expm1(-lambda * (b - a)) / lambda;
}

//! Number of atoms of the last nuclide on a linear decay trajectory at the end
//! of the history, per unit production rate of the first nuclide at the final
//! power and per unit product of branching ratios along the trajectory
//!
//! \param lambda Decay constants of the nuclides on the trajectory in [1/s]
//! \param h Irradiation history
double bateman(const vector<double>& lambda, const History& h)
{
  int n = lambda.size();
  double result = 0.0;
  for (int k = 0; k < n; ++k) {
    // Coefficient of the k-th exponential. The decay constants of the parents
    // in the numerator are paired with the differences in the denominator to
    // avoid overflow on long trajectories.
    double coeff = 1.0;
    int j = 0;
    for (int m = 0; m < n; ++m) {
      if (m == k)
        continue;
      coeff *= lambda[j++] / (lambda[m] - lambda[k]);
    }

    // Integrate the response over the history
    double f = 0.0;
    for (size_t s = 0; s < h.power.size(); ++s) {
      if (h.power[s] > 0.0) {
        f += h.power[s] * exp_integral(lambda[k], h.end[s], h.start[s]);
      }
    }
    result += coeff * f;
  }
  return result;
}

//! Shift a decay constant away from those already on a trajectory
double separate_decay_constant(double lambda, const vector<double>& others)
{
  bool shifted = true;
  while (shifted) {
    shifted = false;
    for (double other : others) {
      if (std::abs(lambda - other) <=
          LAMBDA_SEPARATION * std::max(lambda, other)) {
        lambda = other * (1.0 + 2.0 * LAMBDA_SEPARATION);
        shifted = true;
      }
    }
  }
  return lambda;
}

//! Find the reactions of a nuclide whose cross sections give the cross
//! section of a reaction listed in the depletion chain
vector<int> find_reactions(const Nuclide& nuc, int mt)
{
  if (mt < nuc.reaction_index_.size() && nuc.reaction_index_[mt] != C_NONE) {
    return {static_cast<int>(nuc.reaction_index_[mt])};
  }

  // If the total reaction is not present, sum over the partial reactions to
  // levels of the residual nucleus
  int first, last;
  switch (mt) {
  case N_2N:
    first = N_2N0;
    last = N_2NC;
    break;
  case N_P:
    first = N_P0;
    last = N_PC;
    break;
  case N_D:
    first = N_D0;
    last = N_DC;
    break;
  case N_T:
    first = N_T0;
    last = N_TC;
    break;
  case N_3HE:
    first = N_3HE0;
    last = N_3HEC;
    break;
  case N_A:
    first = N_A0;
    last = N_AC;
    break;
  default:
    return {};
  }
  vector<int> reactions;
  for (int partial = first; partial <= last; ++partial) {
    if (nuc.reaction_index_[partial] != C_NONE) {
      reactions.push_back(nuc.reaction_index_[partial]);
    }
  }
  return reactions;
}

//! Create a channel from the radionuclides left behind by one reaction event
//! \param inventory Atoms of each radionuclide per unit reaction rate [s]
//! \return Channel with emission yields and sampling distributions
DecaySourceChannel make_channel(const std::map<int, double>& inventory)
{
  const auto& s = settings::decay_source_settings;
  DecaySourceChannel channel;
  for (int t = 0; t < N_DECAY_PARTICLES; ++t) {
    if (!s.particles[t])
      continue;

    vector<double> rates;
    for (const auto& [i_chain, atoms] : inventory) {
      const auto* dist =
        data::chain_nuclides[i_chain]->source(DECAY_PARTICLES[t]);
      if (!dist)
        continue;
      // The source integral is the emission rate in [particles/s/atom]
      double rate = atoms * dist->integral();
      if (rate > 0.0) {
        channel.nuclides[t].push_back(i_chain);
        rates.push_back(rate);
      }
    }
    if (!rates.empty()) {
      channel.nuclide_dist[t].assign(rates);
      channel.yield[t] = channel.nuclide_dist[t].integral();
    }
  }
  return channel;
}

bool has_yield(const DecaySourceChannel& channel)
{
  return std::any_of(channel.yield.begin(), channel.yield.end(),
    [](double y) { return y > 0.0; });
}

//! Cross section of an activation channel, including the eigenvalue weighting
//! of non-fission secondary particles
double activation_rate(const DecaySourceChannel& channel, const Nuclide& nuc,
  const Particle& p, int i_nuclide)
{
  const auto& micro = p.neutron_xs(i_nuclide);
  if (micro.index_temp < 0)
    return 0.0;
  double xs = 0.0;
  for (int i_rx : channel.reactions) {
    xs += nuc.reactions_[i_rx]->xs(micro);
  }

  // As for prompt photons from non-fission reactions, increase the weight in
  // k-eigenvalue calculations to properly balance energy release and
  // deposition. See D. P. Griesheimer, S. J. Douglass, and M. H. Stedry,
  // "Self-consistent energy normalization for quasistatic reactor
  // calculations", Proc. PHYSOR, Cambridge, UK, Mar 29-Apr 2, 2020.
  if (settings::run_mode == RunMode::EIGENVALUE) {
    xs *= simulation::keff;
  }
  return xs;
}

//! Interpolation factors of the fission channels at the incident energy
//! \return Index of the lower channel and the interpolation factor
std::pair<int, double> fission_interpolation(
  const vector<DecaySourceChannel>& fission, double E)
{
  int n = fission.size();
  if (n == 1 || E <= fission.front().energy)
    return {0, 0.0};
  if (E >= fission.back().energy)
    return {n - 1, 0.0};
  int i = 0;
  while (fission[i + 1].energy < E) {
    ++i;
  }
  double f =
    (E - fission[i].energy) / (fission[i + 1].energy - fission[i].energy);
  return {i, f};
}

} // namespace

//==============================================================================
// Non-member functions
//==============================================================================

void read_decay_source_settings(pugi::xml_node node)
{
  settings::decay_sources_on = true;
  auto& s = settings::decay_source_settings;
  s = DecaySourceSettings {};

  if (check_for_node(node, "timesteps")) {
    s.timesteps = get_node_array<double>(node, "timesteps");
    s.power.assign(s.timesteps.size(), 1.0);
  }
  if (check_for_node(node, "power")) {
    s.power = get_node_array<double>(node, "power");
  }

  if (s.timesteps.empty()) {
    fatal_error("At least one timestep must be given for decay sources.");
  }
  if (s.power.size() != s.timesteps.size()) {
    fatal_error(fmt::format("Number of decay source timesteps ({}) does not "
                            "match the number of power values ({}).",
      s.timesteps.size(), s.power.size()));
  }
  for (size_t i = 0; i < s.timesteps.size(); ++i) {
    if (s.timesteps[i] <= 0.0) {
      fatal_error("Decay source timesteps must be positive.");
    }
    if (s.power[i] < 0.0) {
      fatal_error("Decay source power values must not be negative.");
    }
  }
  if (s.power.back() <= 0.0) {
    fatal_error("The power at the end of the decay source history must be "
                "positive since it corresponds to the simulated state.");
  }

  if (check_for_node(node, "particles")) {
    s.particles.fill(false);
    for (const auto& name : get_node_array<std::string>(node, "particles")) {
      auto type = ParticleType(name);
      auto it = std::find(DECAY_PARTICLES.begin(), DECAY_PARTICLES.end(), type);
      if (it == DECAY_PARTICLES.end()) {
        fatal_error(fmt::format(
          "Decay sources cannot emit particles of type '{}'.", name));
      }
      s.particles[it - DECAY_PARTICLES.begin()] = true;
    }
  }
}

std::unordered_map<int, double> decay_inventory(
  int i_chain, const DecaySourceSettings& s)
{
  std::unordered_map<int, double> inventory;
  double lambda = data::chain_nuclides[i_chain]->decay_constant();
  if (lambda == 0.0)
    return inventory;

  History h = make_history(s);

  // Node of a trajectory being followed
  struct Node {
    vector<int> nuclides;   //!< Chain indices of nuclides on trajectory
    vector<double> lambda;  //!< Decay constants [1/s]
    double branching_ratio; //!< Product of branching ratios
  };

  vector<Node> stack {{{i_chain}, {lambda}, 1.0}};
  while (!stack.empty()) {
    Node node = std::move(stack.back());
    stack.pop_back();

    // Atoms of the last nuclide on the trajectory at the end of the history
    double atoms = node.branching_ratio * bateman(node.lambda, h);
    inventory[node.nuclides.back()] += std::max(atoms, 0.0);

    // Atoms that have decayed out of the last nuclide, obtained as the
    // inventory of a fictitious stable daughter
    auto lambda_passed = node.lambda;
    lambda_passed.push_back(0.0);
    double passed = node.branching_ratio * bateman(lambda_passed, h);
    if (passed < TTA_CUTOFF * h.produced)
      continue;
    if (node.nuclides.size() >= TTA_MAX_DEPTH) {
      warning(fmt::format("Decay trajectory starting from {} exceeds {} "
                          "nuclides and was truncated.",
        data::chain_nuclides[i_chain]->name(), TTA_MAX_DEPTH));
      continue;
    }

    // Follow each decay mode leading to a radionuclide
    const auto& parent = *data::chain_nuclides[node.nuclides.back()];
    for (const auto& product : parent.decay_products()) {
      auto it = data::chain_nuclide_map.find(product.name);
      if (it == data::chain_nuclide_map.end())
        continue;
      int i_daughter = it->second;
      double lambda_daughter =
        data::chain_nuclides[i_daughter]->decay_constant();
      if (lambda_daughter == 0.0)
        continue;

      // Guard against cycles in malformed chain data
      if (std::find(node.nuclides.begin(), node.nuclides.end(), i_daughter) !=
          node.nuclides.end())
        continue;

      Node daughter {node.nuclides, node.lambda,
        node.branching_ratio * product.branching_ratio};
      daughter.nuclides.push_back(i_daughter);
      daughter.lambda.push_back(
        separate_decay_constant(lambda_daughter, node.lambda));
      stack.push_back(std::move(daughter));
    }
  }
  return inventory;
}

bool decay_sources_produce_fission_photons(const std::string& name)
{
  if (!settings::decay_sources_on ||
      !settings::decay_source_settings.particles[0])
    return false;
  auto it = data::chain_nuclide_map.find(name);
  if (it == data::chain_nuclide_map.end())
    return false;
  return !data::chain_nuclides[it->second]->fission_yields().empty();
}

void finalize_decay_sources()
{
  data::decay_sources.clear();
  if (!settings::decay_sources_on)
    return;

  if (data::chain_nuclides.empty()) {
    fatal_error("Decay sources require a depletion chain with decay data. Set "
                "openmc.config['chain_file'] or the OPENMC_CHAIN_FILE "
                "environment variable.");
  }

  write_message("Computing steady-state decay sources...", 6);

  const auto& s = settings::decay_source_settings;
  data::decay_sources.resize(data::nuclides.size());

  // Inventories left behind by production of each radionuclide are shared
  // among channels, e.g., fission products of different fissionable nuclides
  std::unordered_map<int, std::unordered_map<int, double>> cache;
  auto add_inventory = [&](std::map<int, double>& inventory,
                         const std::string& name, double amount) {
    auto it = data::chain_nuclide_map.find(name);
    if (it == data::chain_nuclide_map.end() || amount <= 0.0)
      return;
    int i_chain = it->second;
    auto cached = cache.find(i_chain);
    if (cached == cache.end()) {
      cached = cache.emplace(i_chain, decay_inventory(i_chain, s)).first;
    }
    for (const auto& [j_chain, atoms] : cached->second) {
      inventory[j_chain] += amount * atoms;
    }
  };

  int n_with_sources = 0;
  for (int i = 0; i < data::nuclides.size(); ++i) {
    const auto& nuc = *data::nuclides[i];
    auto& source = data::decay_sources[i];
    auto it = data::chain_nuclide_map.find(nuc.name_);
    if (it == data::chain_nuclide_map.end()) {
      if (nuc.fissionable_ && s.particles[0] && !nuc.delayed_photon_scaling_) {
        warning(fmt::format("{} is not in the depletion chain. Delayed photons "
                            "from its fission will not be produced.",
          nuc.name_));
      }
      continue;
    }
    const auto& chain_nuc = *data::chain_nuclides[it->second];

    // Activation channels, sorted by MT for reproducibility
    std::map<int, const vector<ChainNuclide::Product>*> products_by_mt;
    for (const auto& [mt, products] : chain_nuc.reaction_products()) {
      if (!is_fission(mt))
        products_by_mt[mt] = &products;
    }
    for (const auto& [mt, products] : products_by_mt) {
      auto reactions = find_reactions(nuc, mt);
      if (reactions.empty())
        continue;
      std::map<int, double> inventory;
      for (const auto& product : *products) {
        add_inventory(inventory, product.name, product.branching_ratio);
      }
      auto channel = make_channel(inventory);
      if (has_yield(channel)) {
        channel.mt = mt;
        channel.reactions = std::move(reactions);
        source.activation.push_back(std::move(channel));
      }
    }

    // Fission product channels at each incident energy with yields
    if (nuc.fissionable_) {
      const auto& all_yields = chain_nuc.fission_yields();
      if (all_yields.empty() && s.particles[0] &&
          !nuc.delayed_photon_scaling_) {
        warning(fmt::format("{} has no fission product yields in the "
                            "depletion chain. Delayed photons from its fission "
                            "will not be produced.",
          nuc.name_));
      }
      for (const auto& yields : all_yields) {
        std::map<int, double> inventory;
        for (const auto& product : yields.products) {
          add_inventory(inventory, product.name, product.branching_ratio);
        }
        auto channel = make_channel(inventory);
        channel.fission = true;
        channel.mt = N_FISSION;
        channel.energy = yields.energy;
        source.fission.push_back(std::move(channel));
      }
      std::sort(source.fission.begin(), source.fission.end(),
        [](const auto& a, const auto& b) { return a.energy < b.energy; });
      source.fission_electrons = std::any_of(source.fission.begin(),
        source.fission.end(), [](const auto& c) { return c.yield[1] > 0.0; });
    }

    if (!source.empty())
      ++n_with_sources;
  }

  write_message(
    6, "Decay sources produced by reactions of {} nuclides", n_with_sources);
}

void sample_decay_sources(Particle& p, int i_nuclide)
{
  // Nuclides loaded after initialization have no decay sources
  if (i_nuclide >= data::decay_sources.size())
    return;
  const auto& source = data::decay_sources[i_nuclide];
  if (source.empty())
    return;

  const auto& nuc = *data::nuclides[i_nuclide];
  const auto& micro = p.neutron_xs(i_nuclide);
  if (micro.total <= 0.0)
    return;

  // Determine the fission channels bracketing the incident energy
  int i_fission = 0;
  double f = 0.0;
  if (!source.fission.empty()) {
    std::tie(i_fission, f) = fission_interpolation(source.fission, p.E());
  }

  // Reaction rate of each channel per unit atom density and flux
  auto channel_rate = [&](const DecaySourceChannel& channel, int i) {
    if (channel.fission) {
      if (i == i_fission)
        return (1.0 - f) * micro.fission;
      if (i == i_fission + 1)
        return f * micro.fission;
      return 0.0;
    }
    return activation_rate(channel, nuc, p, i_nuclide);
  };

  for (int t = 0; t < N_DECAY_PARTICLES; ++t) {
    // Determine the total production cross section of this particle type
    double total = 0.0;
    for (int i = 0; i < source.fission.size(); ++i) {
      total += channel_rate(source.fission[i], i) * source.fission[i].yield[t];
    }
    for (const auto& channel : source.activation) {
      total += channel_rate(channel, -1) * channel.yield[t];
    }
    if (total <= 0.0)
      continue;

    // Sample the number of particles with weight equal to the neutron's weight
    double y_t = total / micro.total;
    int n = static_cast<int>(y_t);
    if (prn(p.current_seed()) <= y_t - n)
      ++n;

    for (int k = 0; k < n; ++k) {
      // Sample the channel. The last channel with a nonzero contribution is
      // kept to guard against round-off in the cumulative sum.
      const DecaySourceChannel* channel = nullptr;
      double cutoff = prn(p.current_seed()) * total;
      double prob = 0.0;
      auto check_channel = [&](const DecaySourceChannel& c, int i) {
        double contribution = channel_rate(c, i) * c.yield[t];
        if (contribution <= 0.0)
          return false;
        channel = &c;
        prob += contribution;
        return prob > cutoff;
      };
      bool found = false;
      for (int i = 0; i < source.fission.size() && !found; ++i) {
        found = check_channel(source.fission[i], i);
      }
      for (int i = 0; i < source.activation.size() && !found; ++i) {
        found = check_channel(source.activation[i], -1);
      }

      // Sample the emitting radionuclide, then the energy from its spectrum
      int i_chain =
        channel->nuclides[t][channel->nuclide_dist[t].sample(p.current_seed())];
      const auto* dist =
        data::chain_nuclides[i_chain]->source(DECAY_PARTICLES[t]);
      auto [E, wgt] = dist->sample(p.current_seed());

      // Decay particles are emitted isotropically
      Direction u = isotropic_direction(p.current_seed());
      if (p.create_secondary(p.wgt() * wgt, u, E, DECAY_PARTICLES[t])) {
        p.local_secondary_bank().back().parent_nuclide = i_chain;
      }
    }
  }
}

double decay_source_beta_heating(const Particle& p, int i_nuclide)
{
  if (!settings::decay_sources_on || i_nuclide >= data::decay_sources.size() ||
      !data::decay_sources[i_nuclide].fission_electrons)
    return 0.0;
  const auto& nuc = *data::nuclides[i_nuclide];
  if (!nuc.betas_)
    return 0.0;
  return (*nuc.betas_)(p.E_last()) * p.neutron_xs(i_nuclide).fission;
}

void free_memory_decay_sources()
{
  data::decay_sources.clear();
}

} // namespace openmc
