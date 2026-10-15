//! \file decay_source.cpp
//! \brief Steady-state sources of decay photons, electrons, and positrons

#include "openmc/decay_source.h"

#include <algorithm> // for sort, any_of
#include <cmath>     // for exp, expm1
#include <map>
#include <numeric> // for accumulate
#include <utility> // for pair

#include <fmt/core.h>

#include "openmc/chain.h"
#include "openmc/constants.h"
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
DecayHistory decay_history;
vector<DecayTrajectory> decay_trajectories;

} // namespace data

//==============================================================================
// DecayHistory implementation
//==============================================================================

DecayHistory::DecayHistory(const DecaySourceSettings& s)
  : infinite {s.timesteps.empty()}, half_life_cutoff {s.half_life_cutoff}
{
  // An infinitely long irradiation at constant power is represented by a
  // single step extending infinitely far into the past
  if (infinite) {
    power = {1.0};
    start = {INFTY};
    end = {0.0};
    produced = INFTY;
    return;
  }

  double total_time = 0.0;
  for (double dt : s.timesteps) {
    total_time += dt;
  }

  double final_power = s.power.back();
  double t = 0.0;
  for (size_t i = 0; i < s.timesteps.size(); ++i) {
    double p = s.power[i] / final_power;
    power.push_back(p);
    start.push_back(total_time - t);
    t += s.timesteps[i];
    end.push_back(total_time - t);
    produced += p * s.timesteps[i];
  }
}

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

//! Integral of exp(-lambda*t) for t from a to b
double exp_integral(double lambda, double a, double b)
{
  if (lambda == 0.0)
    return b - a;
  return std::exp(-lambda * a) * -std::expm1(-lambda * (b - a)) / lambda;
}

//! Coefficients of the exponentials in the Bateman solution for the last
//! nuclide on a linear decay trajectory produced at unit rate. The response to
//! production at time zero is the sum of coeff[k]*exp(-lambda[k]*t).
//!
//! \param lambda Decay constants of the nuclides on the trajectory in [1/s]
vector<double> bateman_coefficients(const vector<double>& lambda)
{
  int n = lambda.size();
  vector<double> coeff(n, 1.0);
  for (int k = 0; k < n; ++k) {
    // The decay constants of the parents in the numerator are paired with the
    // differences in the denominator to avoid overflow on long trajectories
    int j = 0;
    for (int m = 0; m < n; ++m) {
      if (m == k)
        continue;
      coeff[k] *= lambda[j++] / (lambda[m] - lambda[k]);
    }
  }
  return coeff;
}

//! Integral of the Bateman response for delays from a to b
double bateman_integral(
  const vector<double>& lambda, const vector<double>& coeff, double a, double b)
{
  double result = 0.0;
  for (int k = 0; k < lambda.size(); ++k) {
    result += coeff[k] * exp_integral(lambda[k], a, b);
  }
  return result;
}

//! Number of atoms of the last nuclide on a linear decay trajectory at the end
//! of the history that were produced during each step, per unit production
//! rate of the first nuclide at the final power and per unit product of
//! branching ratios along the trajectory
//!
//! \param lambda Decay constants of the nuclides on the trajectory in [1/s]
//! \param h Irradiation history
vector<double> bateman_steps(
  const vector<double>& lambda, const DecayHistory& h)
{
  auto coeff = bateman_coefficients(lambda);
  vector<double> atoms(h.power.size(), 0.0);
  for (size_t s = 0; s < h.power.size(); ++s) {
    if (h.power[s] > 0.0) {
      atoms[s] = std::max(
        h.power[s] * bateman_integral(lambda, coeff, h.end[s], h.start[s]),
        0.0);
    }
  }
  return atoms;
}

//! Total over the history of bateman_steps
double bateman(const vector<double>& lambda, const DecayHistory& h)
{
  auto atoms = bateman_steps(lambda, h);
  return std::accumulate(atoms.begin(), atoms.end(), 0.0);
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

//! Decay constant of a chain nuclide, treating nuclides with half-lives above
//! the cutoff as stable
double decay_constant(int i_chain, const DecayHistory& h)
{
  double lambda = data::chain_nuclides[i_chain]->decay_constant();
  if (lambda > 0.0 && std::log(2.0) / lambda > h.half_life_cutoff)
    return 0.0;
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
//! \param production Amount of each decay trajectory per reaction, indexed by
//!   position in data::decay_trajectories
//! \return Channel with emission yields and sampling distributions
DecaySourceChannel make_channel(const std::map<int, double>& production)
{
  const auto& s = settings::decay_source_settings;
  DecaySourceChannel channel;
  for (int t = 0; t < N_DECAY_PARTICLES; ++t) {
    if (!s.particles[t])
      continue;

    vector<double> rates;
    for (const auto& [i_traj, amount] : production) {
      const auto& traj = data::decay_trajectories[i_traj];
      const auto* dist =
        data::chain_nuclides[traj.nuclide]->source(DECAY_PARTICLES[t]);
      if (!dist)
        continue;
      // The source integral is the emission rate in [particles/s/atom]
      double rate = amount * traj.atoms * dist->integral();
      if (rate > 0.0) {
        channel.trajectories[t].push_back(i_traj);
        rates.push_back(rate);
        channel.decay_energy[t] +=
          amount * traj.atoms *
          data::chain_nuclides[traj.nuclide]->source_energy(DECAY_PARTICLES[t]);
      }
    }
    if (!rates.empty()) {
      channel.trajectory_dist[t].assign(rates);
      channel.yield[t] = channel.trajectory_dist[t].integral();
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

//! Disable electron and positron emission if the chain represents beta
//! transitions by their endpoint energies, as in chain files generated with
//! older versions of OpenMC. Such data are detected because the energy of
//! decay electrons and positrons, including annihilation, exceeds the total
//! decay energy of many nuclides.
void check_charged_particle_sources()
{
  auto& s = settings::decay_source_settings;
  if (!s.particles[1] && !s.particles[2])
    return;

  int n_checked = 0;
  int n_inconsistent = 0;
  for (const auto& nuc : data::chain_nuclides) {
    double lambda = nuc->decay_constant();
    double e = nuc->source_energy(ParticleType::electron()) +
               nuc->source_energy(ParticleType::positron());
    const auto* positron = nuc->source(ParticleType::positron());
    if (positron) {
      e += 2.0 * MASS_ELECTRON_EV * positron->integral();
    }
    if (lambda == 0.0 || nuc->decay_energy() <= 0.0 || e == 0.0)
      continue;
    ++n_checked;
    if (e / lambda > 1.05 * nuc->decay_energy() + 1.0e3)
      ++n_inconsistent;
  }

  if (n_inconsistent > 0.1 * n_checked) {
    warning(fmt::format(
      "The electron and positron decay sources in the depletion chain exceed "
      "the decay energy of {} of {} nuclides, which indicates that beta "
      "transitions are represented by their endpoint energies. Only photons "
      "will be produced by decay sources. Regenerate the chain with "
      "openmc.deplete.Chain.from_endf to include electrons and positrons.",
      n_inconsistent, n_checked));
    s.particles[1] = false;
    s.particles[2] = false;
  }
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

  // Without timesteps, the irradiation at constant power is infinitely long
  if (check_for_node(node, "timesteps")) {
    s.timesteps = get_node_array<double>(node, "timesteps");
    s.power.assign(s.timesteps.size(), 1.0);
    if (s.timesteps.empty()) {
      fatal_error("At least one timestep must be given for decay sources.");
    }
  }
  if (check_for_node(node, "power")) {
    if (s.timesteps.empty()) {
      fatal_error("Decay source power values require timesteps.");
    }
    s.power = get_node_array<double>(node, "power");
  }
  if (check_for_node(node, "half_life_cutoff")) {
    s.half_life_cutoff = std::stod(get_node_value(node, "half_life_cutoff"));
    if (s.half_life_cutoff <= 0.0) {
      fatal_error("Decay source half-life cutoff must be positive.");
    }
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
  if (!s.power.empty() && s.power.back() <= 0.0) {
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

  // Read the decay data needed from the depletion chain
  read_chain_decay_source_data();
  check_charged_particle_sources();
}

vector<DecayTrajectory> find_decay_trajectories(
  int i_chain, const DecayHistory& h)
{
  vector<DecayTrajectory> trajectories;
  double lambda = decay_constant(i_chain, h);
  if (lambda == 0.0)
    return trajectories;

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
    DecayTrajectory traj;
    traj.nuclide = node.nuclides.back();
    traj.branching_ratio = node.branching_ratio;
    traj.lambda = node.lambda;
    traj.step_atoms = bateman_steps(node.lambda, h);
    for (auto& atoms : traj.step_atoms) {
      atoms *= node.branching_ratio;
    }
    traj.atoms =
      std::accumulate(traj.step_atoms.begin(), traj.step_atoms.end(), 0.0);
    if (traj.atoms > 0.0) {
      trajectories.push_back(std::move(traj));
    }

    // Fraction of the produced atoms that have decayed out of the last
    // nuclide, obtained from the inventory of a fictitious stable daughter.
    // With an infinitely long irradiation, all atoms have decayed out.
    double passed = node.branching_ratio;
    if (!h.infinite) {
      auto lambda_passed = node.lambda;
      lambda_passed.push_back(0.0);
      passed *= bateman(lambda_passed, h) / h.produced;
    }
    if (passed < TTA_CUTOFF)
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
      double lambda_daughter = decay_constant(i_daughter, h);
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
  return trajectories;
}

std::unordered_map<int, double> decay_inventory(
  int i_chain, const DecaySourceSettings& s)
{
  std::unordered_map<int, double> inventory;
  for (const auto& traj : find_decay_trajectories(i_chain, DecayHistory(s))) {
    inventory[traj.nuclide] += traj.atoms;
  }
  return inventory;
}

double sample_decay_delay(
  const DecayTrajectory& traj, const DecayHistory& h, uint64_t* seed)
{
  // With an infinitely long irradiation, the delay is the sum of the times
  // spent in each nuclide on the trajectory, which are exponentially
  // distributed
  if (h.infinite) {
    double delay = 0.0;
    for (double lambda : traj.lambda) {
      delay -= std::log(prn(seed)) / lambda;
    }
    return delay;
  }

  // Sample the step of the history during which the atom was produced
  double cutoff = prn(seed) * traj.atoms;
  double prob = 0.0;
  int s = 0;
  for (int i = 0; i < traj.step_atoms.size(); ++i) {
    if (traj.step_atoms[i] <= 0.0)
      continue;
    s = i;
    prob += traj.step_atoms[i];
    if (prob > cutoff)
      break;
  }
  double a = h.end[s];
  double b = h.start[s];
  double xi = prn(seed);

  // With a single nuclide, the delay follows a truncated exponential
  const auto& lambda = traj.lambda;
  if (lambda.size() == 1) {
    return a - std::log1p(xi * std::expm1(-lambda[0] * (b - a))) / lambda[0];
  }

  // Otherwise, invert the cumulative distribution of the Bateman response
  // within the step with Newton's method safeguarded by bisection
  auto coeff = bateman_coefficients(lambda);
  double target = xi * bateman_integral(lambda, coeff, a, b);
  double lo = a;
  double hi = b;
  double t = 0.5 * (a + b);
  for (int iter = 0; iter < 100; ++iter) {
    double f = bateman_integral(lambda, coeff, a, t) - target;
    if (f > 0.0) {
      hi = t;
    } else {
      lo = t;
    }
    double df = 0.0;
    for (int k = 0; k < lambda.size(); ++k) {
      df += coeff[k] * std::exp(-lambda[k] * t);
    }
    double t_new = (df > 0.0) ? t - f / df : 0.5 * (lo + hi);
    if (!(t_new > lo && t_new < hi)) {
      t_new = 0.5 * (lo + hi);
    }
    if (std::abs(t_new - t) <= 1.0e-10 * t_new) {
      return t_new;
    }
    t = t_new;
  }
  return t;
}

bool decay_sources_produce_fission_photons(const std::string& name)
{
  if (!settings::decay_sources_on ||
      !settings::decay_source_settings.particles[0])
    return false;
  auto it = data::chain_nuclide_map.find(name);
  if (it == data::chain_nuclide_map.end())
    return false;

  // Fission products need decay photon data, which is checked for the
  // nuclides produced directly
  for (const auto& yields :
    data::chain_nuclides[it->second]->fission_yields()) {
    for (const auto& product : yields.products) {
      auto jt = data::chain_nuclide_map.find(product.name);
      if (jt != data::chain_nuclide_map.end() &&
          data::chain_nuclides[jt->second]->photon_energy()) {
        return true;
      }
    }
  }
  return false;
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
  data::decay_history = DecayHistory(s);
  data::decay_trajectories.clear();

  // Trajectories starting from each produced radionuclide are shared among
  // channels, e.g., fission products of different fissionable nuclides
  std::unordered_map<int, vector<int>> cache;
  auto add_inventory = [&](std::map<int, double>& production,
                         const std::string& name, double amount) {
    auto it = data::chain_nuclide_map.find(name);
    if (it == data::chain_nuclide_map.end() || amount <= 0.0)
      return;
    int i_chain = it->second;
    auto cached = cache.find(i_chain);
    if (cached == cache.end()) {
      vector<int> indices;
      for (auto& traj : find_decay_trajectories(i_chain, data::decay_history)) {
        indices.push_back(data::decay_trajectories.size());
        data::decay_trajectories.push_back(std::move(traj));
      }
      cached = cache.emplace(i_chain, std::move(indices)).first;
    }
    for (int i_traj : cached->second) {
      production[i_traj] += amount;
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
      for (const auto& channel : source.fission) {
        source.fission_photons |= channel.yield[0] > 0.0;
        source.fission_electrons |= channel.yield[1] > 0.0;
      }
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

      // Sample the decay trajectory, then the energy from the spectrum of the
      // emitting radionuclide
      const auto& traj = data::decay_trajectories
        [channel->trajectories[t][channel->trajectory_dist[t].sample(
          p.current_seed())]];
      const auto* dist =
        data::chain_nuclides[traj.nuclide]->source(DECAY_PARTICLES[t]);
      auto [E, wgt] = dist->sample(p.current_seed());

      // Decay particles are emitted isotropically after the delay between
      // the reaction and the decay, as for delayed neutrons
      Direction u = isotropic_direction(p.current_seed());
      double delay =
        sample_decay_delay(traj, data::decay_history, p.current_seed());
      if (p.create_secondary(p.wgt() * wgt, u, E, DECAY_PARTICLES[t])) {
        auto& site = p.local_secondary_bank().back();
        site.parent_nuclide = traj.nuclide;
        site.time += delay;
      }
    }
  }
}

DecayHeating decay_source_heating(const Particle& p, int i_nuclide, int mt)
{
  DecayHeating heating;
  if (!settings::decay_sources_on || i_nuclide >= data::decay_sources.size())
    return heating;
  const auto& source = data::decay_sources[i_nuclide];
  if (source.empty())
    return heating;

  // With photon transport, decay particles are emitted and deposit their own
  // energy, so 'heating-local' is not modified
  bool local = !settings::photon_transport;
  if (!local && mt != HEATING)
    return heating;

  // Kinetic energy of decay particles deposited locally. Without photon
  // transport, 'heating' includes only charged particles while
  // 'heating-local' also includes photons and the annihilation of positrons.
  bool include_photons = local && mt == HEATING_LOCAL;
  auto deposited = [&](const DecaySourceChannel& c) {
    if (!local)
      return 0.0;
    double e = c.decay_energy[1] + c.decay_energy[2];
    if (include_photons) {
      e += c.decay_energy[0] + 2.0 * MASS_ELECTRON_EV * c.yield[2];
    }
    return e;
  };

  const auto& nuc = *data::nuclides[i_nuclide];
  const auto& micro = p.neutron_xs(i_nuclide);
  double E = p.E_last();

  if (!source.fission.empty()) {
    // Remove the delayed energy from the fission heating number for the
    // particle types produced by decay sources
    if (nuc.betas_ && source.fission_electrons) {
      heating.fission -= (*nuc.betas_)(E)*micro.fission;
    }
    if (include_photons && nuc.delayed_photons_ && source.fission_photons) {
      heating.fission -= (*nuc.delayed_photons_)(E)*micro.fission;
    }

    // Add the energy of decay particles deposited locally
    if (local) {
      auto [i, f] = fission_interpolation(source.fission, E);
      double e = (1.0 - f) * deposited(source.fission[i]);
      if (f > 0.0)
        e += f * deposited(source.fission[i + 1]);
      heating.fission += e * micro.fission;
    }
  }

  // Add the energy of decay particles from activation products deposited
  // locally. Weighting by keff is applied with the other non-fission heating.
  if (local && micro.index_temp >= 0) {
    for (const auto& channel : source.activation) {
      double xs = 0.0;
      for (int i_rx : channel.reactions) {
        xs += nuc.reactions_[i_rx]->xs(micro);
      }
      heating.activation += xs * deposited(channel);
    }
  }
  return heating;
}

void free_memory_decay_sources()
{
  data::decay_sources.clear();
  data::decay_trajectories.clear();
  data::decay_history = DecayHistory {};
}

} // namespace openmc
