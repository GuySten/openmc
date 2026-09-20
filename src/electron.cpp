#include "openmc/photon.h"

#include "openmc/array.h"
#include "openmc/bremsstrahlung.h"
#include "openmc/condensed_history.h"
#include "openmc/constants.h"
#include "openmc/gos.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/particle.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/settings.h"

#include "openmc/tensor.h"

#include <algorithm> // for max, min
#include <cmath>
#include <fmt/core.h>

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

//==============================================================================
// Electron interaction data, read into Element
//==============================================================================

namespace {

//! Put a quantity tabulated on the partial-wave energy grid onto the (denser)
//! electron grid, interpolating log-log.
//!
//! Both transport moments fall as a power of the energy across each interval
//! of the geometric partial-wave grid, so a linear blend across an interval
//! spanning a factor of 1.2 is a chord across a convex curve. This is the same
//! argument that makes the angular distributions themselves log-log
//! interpolated in energy.
tensor::Tensor<double> moments_on_grid(const tensor::Tensor<double>& grid,
  const vector<double>& energy, const vector<double>& value)
{
  auto n = grid.size();
  auto out = tensor::zeros<double>({n});
  if (energy.size() < 2)
    return out;

  for (decltype(n) i = 0; i < n; ++i) {
    double E = grid(i);
    int k = lower_bound_index(energy.begin(), energy.end(), E);
    k = std::max(0, std::min(k, static_cast<int>(energy.size()) - 2));

    double v0 = value[k];
    double v1 = value[k + 1];
    double e0 = energy[k];
    double e1 = energy[k + 1];
    if (v0 > 0.0 && v1 > 0.0 && e0 > 0.0 && e1 > e0 && E > 0.0) {
      double f = std::log(E / e0) / std::log(e1 / e0);
      out(i) = std::exp((1.0 - f) * std::log(v0) + f * std::log(v1));
    } else if (e1 > e0) {
      double f = (E - e0) / (e1 - e0);
      out(i) = std::max(0.0, v0 + f * (v1 - v0));
    } else {
      out(i) = v0;
    }
  }
  return out;
}

//! Clamped linear lookup on the electron energy grid
int grid_index(const tensor::Tensor<double>& grid, double E, double& f)
{
  int n = grid.size();
  int i = upper_bound_index(grid.cbegin(), grid.cend(), E);
  i = std::max(0, std::min(i, n - 2));
  f = (E - grid(i)) / (grid(i + 1) - grid(i));
  f = std::max(0.0, std::min(1.0, f));
  return i;
}

} // namespace

void Element::read_electron_data(hid_t group)
{
  // name_, Z_ and index_ are already set from the photoatomic data; only the
  // electron grid and reactions are read here.
  read_dataset(group, "energy", electron_energy_);

  // Read elastic scattering, for both projectile charges. They differ little
  // in rate -- the integrated cross sections agree to under a per cent, which
  // is the Born limit and is symmetric in the charge -- and a great deal in
  // first moment, because a positron is repelled by the nucleus and stays out
  // of the small-impact-parameter region that makes the large deflections.
  hid_t rgroup = open_group(group, "elastic");
  if (attribute_exists(rgroup, "energy_min")) {
    read_attribute(rgroup, "energy_min", elastic_energy_min_);
    read_attribute(rgroup, "energy_max", elastic_energy_max_);
  }
  for (int q = 0; q < 2; ++q) {
    hid_t qgroup = open_group(rgroup, q == 0 ? "electron" : "positron");
    read_dataset(qgroup, "xs", elastic_[q]);
    hid_t qdist = open_group(qgroup, "distribution");
    // Interpolate between the tabulated distributions log-log in energy rather
    // than with the lin_lin rule the data carries. EEDL does specify INT=2 for
    // this TAB2, but that rule assumes the tables are close enough together
    // for a linear blend of them to mean something, and here they are not.
    //
    // The partial-wave grid is geometric, about fifteen points per decade, and
    // 1-<mu> falls as a power of the energy across each interval. A linear
    // blend of two tables a factor of 1.2 apart is a chord across a convex
    // curve, and picking one of them at random -- what the lin_lin rule
    // amounts to -- reproduces the arithmetic mean of the two rather than the
    // power law. Blending the deflection geometrically follows the trend
    // instead: measured against a leave-one-out reconstruction it recovers
    // <1-mu> to a few tenths of a per cent, against a few per cent for the
    // default rule.
    elastic_angle_[q] = AngleDistribution {qdist, Interpolation::log_log};
    close_group(qdist);
    close_group(qgroup);

    // Where the elastic distribution splits into soft and hard, for a mixed
    // condensed-history step, and the transport moments of the soft half.
    // <1-mu> is what sets the scale a step may cover: after a path s the mean
    // deflection is exp(-s n sigma_el <1-mu>), so 1/(n sigma_el <1-mu>) is the
    // first transport mean free path. C1 = 0 is the default and means every
    // collision is hard, so this costs nothing until a run asks for it.
    if (settings::deflection_cutoff > 0.0) {
      vector<double> s_energy, mu_cut, p_hard, m1_soft, m2_soft;
      elastic_angle_[q].restricted_moments(settings::deflection_cutoff,
        s_energy, mu_cut, p_hard, m1_soft, m2_soft);
      elastic_p_hard_[q] = moments_on_grid(electron_energy_, s_energy, p_hard);
      elastic_mu1_soft_[q] =
        moments_on_grid(electron_energy_, s_energy, m1_soft);
      elastic_mu2_soft_[q] =
        moments_on_grid(electron_energy_, s_energy, m2_soft);
    }
  }
  close_group(rgroup);

  // Read ionization
  rgroup = open_group(group, "ionization");
  read_dataset(rgroup, "xs", electroionization_);
  vector<std::string> designators;
  read_attribute(rgroup, "designators", designators);
  close_group(rgroup);

  // Map each electroionization subshell onto the corresponding entry of
  // shells_, which holds the binding energies and relaxation transitions.
  // Matching is by ENDF designator: the electroionization list (NXS(7)
  // subshells) and shells_ need not agree in length or order, and in
  // particular neither corresponds to the Compton Doppler broadening shell
  // list (NXS(5) shells).
  electron_shell_map_.resize(designators.size(), -1);
  for (int i = 0; i < designators.size(); ++i) {
    int endf_index = 0;
    int j = 1;
    for (const auto& subshell : SUBSHELLS) {
      if (designators[i] == subshell) {
        endf_index = j;
        break;
      }
      ++j;
    }

    for (int k = 0; k < shells_.size(); ++k) {
      if (shells_[k].index_subshell == endf_index) {
        electron_shell_map_[i] = k;
        break;
      }
    }

    if (electron_shell_map_[i] < 0) {
      fatal_error(fmt::format(
        "Electroionization subshell {} of element {} has no counterpart in the "
        "photoatomic data, so its binding energy and relaxation transitions "
        "are unavailable. The electron and photon libraries are inconsistent.",
        designators[i], name_));
    }
  }

  // Read bremsstrahlung. Only the cross section is here: it is an integral of
  // the scaled cross sections of the photon library above the threshold stored
  // alongside it, and the emitted photon energy is sampled from that same
  // table. Storing the distribution here as well would be storing the same
  // numbers twice and inviting the rate and the spectrum to drift apart.
  rgroup = open_group(group, "bremsstrahlung");
  read_dataset(rgroup, "xs", electron_bremsstrahlung_);
  read_attribute(rgroup, "photon_cutoff", bremsstrahlung_photon_cutoff_);
  close_group(rgroup);

  // The soft/hard split of the channels the atom owns, which needs every one
  // of them loaded and so comes last. Skipped unless a run asks for condensed
  // history, since it is the only thing that reads it.
  if (settings::deflection_cutoff > 0.0) {
    this->compute_step_tables();
  }
}

namespace {

//! Integral of W^order times the Bhabha cross section shape over W, with the
//! leading constant dropped. Every term is elementary.
//!
//! Order 0 is the cross section, 1 the stopping power it contributes and 2 the
//! straggling. The shape is a Laurent polynomial in x = W/E, so raising the
//! order shifts every term by one and nothing but the leading 1/x^2 ever needs
//! care.
double bhabha_moment(
  const FreeCollision& c, double E, double W_lo, double W_hi, int order)
{
  if (W_hi <= W_lo || E <= 0.0)
    return 0.0;
  double x_lo = W_lo / E;
  double x_hi = W_hi / E;
  double d1 = x_hi - x_lo;
  double d2 = x_hi * x_hi - x_lo * x_lo;
  double d3 = x_hi * x_hi * x_hi - x_lo * x_lo * x_lo;
  double d4 = x_hi * x_hi * x_hi * x_hi - x_lo * x_lo * x_lo * x_lo;
  double d5 = d4 * x_hi + x_lo * x_lo * x_lo * x_lo * d1;
  double log_ratio = std::log(x_hi / x_lo);

  double a;
  if (order == 0) {
    a = 1.0 / x_lo - 1.0 / x_hi - c.b1 * log_ratio + c.b2 * d1 -
        0.5 * c.b3 * d2 + c.b4 * d3 / 3.0;
  } else if (order == 1) {
    a = log_ratio - c.b1 * d1 + 0.5 * c.b2 * d2 - c.b3 * d3 / 3.0 +
        0.25 * c.b4 * d4;
  } else {
    a = d1 - 0.5 * c.b1 * d2 + c.b2 * d3 / 3.0 - 0.25 * c.b3 * d4 +
        0.2 * c.b4 * d5;
  }
  return std::pow(E, order) * a / E;
}

//! The cross section itself, which is the zeroth moment
//!
//! The counterpart of bhabha_moment() for the electron, and elementary in the
//! same way: the shape is 1/x^2 + 1/(1-x)^2 + A - C/(x(1-x)), every term of
//! which integrates in closed form against any power of x. Order 0 is the
//! cross section, 1 the stopping power it contributes and 2 the straggling.
//!
//! \param[in] c Free collision shape at this projectile energy
//! \param[in] E Kinetic energy in [eV]
//! \param[in] W_lo Lower limit on the energy transfer in [eV]
//! \param[in] W_hi Upper limit, at most E/2 where the two electrons become
//!   indistinguishable
//! \param[in] order Power of W the integrand carries
//! \return The integral, in [eV^(order-1)] times the shape's own units
double moller_moment(
  const FreeCollision& c, double E, double W_lo, double W_hi, int order)
{
  if (W_hi <= W_lo || E <= 0.0)
    return 0.0;
  double x_lo = W_lo / E;
  double x_hi = W_hi / E;
  if (!(x_lo > 0.0) || x_hi >= 1.0)
    return 0.0;
  double u_lo = 1.0 - x_lo;
  double u_hi = 1.0 - x_hi;

  // Antiderivative of x^order times the shape, evaluated at both ends
  auto anti = [&](double x, double u) {
    if (order == 0) {
      return -1.0 / x + 1.0 / u + c.amol * x - c.moller_c * std::log(x / u);
    } else if (order == 1) {
      return std::log(x) + 1.0 / u + std::log(u) + 0.5 * c.amol * x * x +
             c.moller_c * std::log(u);
    }
    return x + 1.0 / u + 2.0 * std::log(u) - u + c.amol * x * x * x / 3.0 +
           c.moller_c * (std::log(u) - u);
  };

  return std::pow(E, order) * (anti(x_hi, u_hi) - anti(x_lo, u_lo)) / E;
}

} // namespace

namespace detail {

double bhabha_moment(double E, double W_lo, double W_hi, int order)
{
  return openmc::bhabha_moment(FreeCollision {E}, E, W_lo, W_hi, order);
}

double moller_moment(double E, double W_lo, double W_hi, int order)
{
  return openmc::moller_moment(FreeCollision {E}, E, W_lo, W_hi, order);
}

} // namespace detail

namespace {

//! Integrals of the scaled bremsstrahlung cross section over a range of kappa
//!
//! chi is linear in kappa between tabulated points, which is what the sampler
//! assumes, so each interval integrates in closed form and the rate stays an
//! integral of exactly the distribution that is sampled.
//!
//! \param[in] chi The scaled cross section, already blended across the two
//!   bracketing incident energies
//! \param[in] k_lo, k_hi Range of kappa
//! \param[in] order Power of kappa the integrand carries beyond chi/kappa, so
//!   that order 0 gives the cross section, 1 the radiated energy and 2 its
//!   second moment, each still missing its power of the incident energy
double chi_integral(
  const vector<double>& chi, double k_lo, double k_hi, int order)
{
  const auto& kappa = data::brems_k_grid;
  int n = kappa.size();
  double total = 0.0;
  for (int j = 0; j + 1 < n; ++j) {
    double x1 = std::max(kappa(j), k_lo);
    double x2 = std::min(kappa(j + 1), k_hi);
    if (x2 <= x1)
      continue;

    double h = kappa(j + 1) - kappa(j);
    double b = (chi[j + 1] - chi[j]) / h;
    double a = chi[j] - b * kappa(j);

    // int (a + b k) k^(order-1) dk
    if (order == 0) {
      total += a * std::log(x2 / x1) + b * (x2 - x1);
    } else if (order == 1) {
      total += a * (x2 - x1) + 0.5 * b * (x2 * x2 - x1 * x1);
    } else {
      total +=
        0.5 * a * (x2 * x2 - x1 * x1) + b * (x2 * x2 * x2 - x1 * x1 * x1) / 3.0;
    }
  }
  return total;
}

} // namespace

//! Tabulate the soft/hard split of the channels the atom owns
//!
//! The inelastic collisions are the medium's rather than any atom's -- their
//! oscillator strengths are shares of all its electrons and their resonance
//! energies are fixed by its mean excitation energy -- so Material builds
//! that split from the oscillator model. What is left here is
//! bremsstrahlung, whose photons below the radiative cutoff are grouped into
//! the step, and the assembly of the hard cross section and its majorant from
//! the fractions the atom's channels contribute.
void Element::compute_step_tables()
{
  int n_energy = electron_energy_.size();
  auto shape_1d = std::vector<size_t> {static_cast<size_t>(n_energy)};

  for (int q = 0; q < 2; ++q) {
    brems_soft_s_[q] = tensor::zeros<double>(shape_1d);
    brems_soft_w2_[q] = tensor::zeros<double>(shape_1d);
    brems_p_hard_[q] = tensor::zeros<double>(shape_1d);
    hard_total_[q] = tensor::zeros<double>(shape_1d);
    hard_majorant_[q] = tensor::zeros<double>(shape_1d);
  }

  const auto& T = data::brems_e_grid;
  int n_brems_e = T.size();
  int n_kappa = data::brems_k_grid.size();
  vector<double> chi(n_kappa);

  for (int j = 0; j < n_energy; ++j) {
    double E = electron_energy_(j);

    // The scaled bremsstrahlung cross section, blended across the two
    // bracketing incident energies exactly as the sampler blends it. It does
    // not depend on the projectile charge; the thresholds that cut it do.
    bool have_chi = false;
    if (n_brems_e > 1 && E > bremsstrahlung_photon_cutoff_) {
      int i_brems;
      double f;
      if (E <= T(0)) {
        i_brems = 0;
        f = 0.0;
      } else if (E >= T(n_brems_e - 1)) {
        i_brems = n_brems_e - 2;
        f = 1.0;
      } else {
        i_brems = lower_bound_index(T.cbegin(), T.cend(), E);
        f = std::log(E / T(i_brems)) / std::log(T(i_brems + 1) / T(i_brems));
      }
      for (int k = 0; k < n_kappa; ++k) {
        chi[k] =
          dcs_(i_brems, k) + f * (dcs_(i_brems + 1, k) - dcs_(i_brems, k));
      }
      have_chi = true;
    }

    // Per projectile charge, because the two thresholds differ: a positron's
    // own cutoff bounds how much a grouped event may take from it, and that
    // cutoff is not the electron's.
    for (int q = 0; q < 2; ++q) {
      // The electron library covers exactly two projectiles, so this is
      // where its charge index becomes a particle again
      ParticleType projectile =
        (q == 0) ? ParticleType::electron() : ParticleType::positron();

      // The rate is taken from the library rather than rebuilt here, and only
      // the fraction of it below the cutoff is computed from the scaled cross
      // section, so the total emission rate stays exactly what the
      // single-event transport uses.
      double w_cr = soft_radiative_cutoff(projectile, E);
      double k_min = bremsstrahlung_photon_cutoff_;
      double k_cut = std::min(w_cr, E);
      double total = have_chi ? chi_integral(chi, k_min / E, 1.0, 0) : 0.0;
      if (have_chi && k_cut > k_min && total > 0.0) {
        double kappa_min = k_min / E;
        double kappa_cut = std::min(1.0, k_cut / E);
        double soft = chi_integral(chi, kappa_min, kappa_cut, 0);
        brems_p_hard_[q](j) = std::max(0.0, 1.0 - soft / total);

        double xs = electron_bremsstrahlung_(j);
        if (q == 1)
          xs *= salvat_factor(Z_ * Z_, E);
        brems_soft_s_[q](j) =
          xs * E * chi_integral(chi, kappa_min, kappa_cut, 1) / total;
        brems_soft_w2_[q](j) =
          xs * E * E * chi_integral(chi, kappa_min, kappa_cut, 2) / total;
      } else {
        brems_p_hard_[q](j) = 1.0;
      }

      // The hard cross section at this energy, assembled from the fractions
      // just tabulated. It depends on nothing but the energy and the charge,
      // which is what lets the flight be drawn from a bound on it.
      //
      // In-flight annihilation is never grouped. Its photons carry away more
      // than the positron's kinetic energy, so no bound on the energy the
      // projectile gives up bounds what they can reach, and a grouped
      // annihilation would be an annihilation that did not happen.
      double brems_hard = electron_bremsstrahlung_(j) * brems_p_hard_[q](j);
      if (q == 1)
        brems_hard *= salvat_factor(Z_ * Z_, E);
      hard_total_[q](j) = elastic_[q](j) * elastic_p_hard_[q](j) +
                          (q == 1 ? this->annihilation_xs(E) : 0.0) +
                          brems_hard;
    }
  }

  // An upper bound on the hard cross section over the energies one step can
  // reach, which is what the flight is actually drawn from. The bound only has
  // to hold; a shade of slack costs a few declined interactions and a shade of
  // shortfall would bias the flight, so a small margin is paid here.
  //
  // How far down to look is set by how much energy a step can actually take,
  // which is more than the budget it is allowed to take on average. The
  // sampled loss overshoots its mean, by sqrt(3 var) where the distribution is
  // uniform and by more where it is not, so a scan to E - budget would stop
  // above the energies the step reaches and bound nothing there. The overshoot
  // is bounded, though. Over a step the loss has mean s*S and variance s*Omega
  // for the restricted stopping power S and straggling Omega, so
  //
  //   var/mean = Omega/S = \int W^2 dsigma / \int W dsigma <= w_cc,
  //
  // every transfer in those integrals being under the soft cutoff; and w_cc is
  // itself held under MAX_SOFT_LOSS_SHARE of the budget. Feeding var <= 0.1
  // mean^2 into the two branches of sample_soft_energy_loss() gives 1.55 and
  // 1.65 times the budget respectively, so twice the budget covers both with
  // room to spare, and the scan costs one more grid point either way.
  vector<double> energy(n_energy);
  for (int j = 0; j < n_energy; ++j) {
    energy[j] = electron_energy_(j);
  }
  for (int q = 0; q < 2; ++q) {
    ParticleType projectile =
      (q == 0) ? ParticleType::electron() : ParticleType::positron();
    vector<double> hard(n_energy), lowest(n_energy);
    for (int j = 0; j < n_energy; ++j) {
      hard[j] = hard_total_[q](j);
      lowest[j] = energy[j] - MAX_SOFT_LOSS_OVERSHOOT *
                                soft_loss_budget(projectile, energy[j]);
    }
    vector<double> majorant = step_majorant(energy, hard, lowest);
    for (int j = 0; j < n_energy; ++j) {
      hard_majorant_[q](j) = majorant[j];
    }
  }
}

void Element::calculate_electron_xs(Particle& p) const
{
  // Perform binary search on the element energy grid in order to determine
  // which points to interpolate between
  int n_grid = electron_energy_.size();
  double E = p.E();
  int i_grid;
  if (E <= electron_energy_[0]) {
    i_grid = 0;
  } else if (E > electron_energy_(n_grid - 1)) {
    i_grid = n_grid - 2;
  } else {
    // We use upper_bound_index here because sometimes photons are created with
    // energies that exactly match a grid point
    i_grid =
      upper_bound_index(electron_energy_.cbegin(), electron_energy_.cend(), E);
  }

  // check for case where two energy points are the same
  if (electron_energy_(i_grid) == electron_energy_(i_grid + 1))
    ++i_grid;

  // Calculate interpolation factor, clamped to the tabulated range. Outside it
  // the factor is unbounded, and extrapolating linearly sends the partials
  // negative: below the first grid point the elastic cross section then
  // exceeds the total and the sampler is pinned on the elastic branch for
  // ever. That is reachable whenever the cutoff is left at zero.
  double f = (E - electron_energy_(i_grid)) /
             (electron_energy_(i_grid + 1) - electron_energy_(i_grid));
  f = std::max(0.0, std::min(1.0, f));

  auto& xs {p.electron_xs(index_)};
  xs.index_grid = i_grid;
  xs.interp_factor = f;

  // Calculate microscopic elastic cross section
  int q = p.type().is_positron() ? 1 : 0;
  xs.elastic =
    elastic_[q](i_grid) + f * (elastic_[q](i_grid + 1) - elastic_[q](i_grid));

  // In-flight annihilation is a channel a positron has and an electron does
  // not. Over a whole slowing-down history it is far from rare: about one
  // positron in six started at 21 MeV annihilates before reaching the cutoff.
  xs.annihilation = p.type().is_positron() ? this->annihilation_xs(E) : 0.0;

  // Calculate microscopic bremsstrahlung cross section. A positron radiates
  // less than an electron of the same energy, being repelled by the nucleus
  // rather than attracted to it. The ratio is independent of the emitted
  // photon energy, so it scales the rate here and leaves the spectrum that
  // sample_bremsstrahlung_energy() draws from untouched.
  xs.bremsstrahlung = electron_bremsstrahlung_(i_grid) +
                      f * (electron_bremsstrahlung_(i_grid + 1) -
                            electron_bremsstrahlung_(i_grid));
  if (p.type().is_positron()) {
    xs.bremsstrahlung *= salvat_factor(Z_ * Z_, E);
  }

  // Calculate microscopic total cross section. The inelastic collisions are
  // not here: they come from the material's oscillator model, which needs the
  // mean excitation energy and the density effect of the medium and so cannot
  // be an element's to give. What the element owns are the channels that
  // belong to the atom alone.
  xs.total = xs.elastic + xs.annihilation + xs.bremsstrahlung;

  // Split it into the part a condensed-history step transports one collision
  // at a time and the part it groups. Without condensed history every channel
  // is hard, which is what leaves the transport below untouched.
  if (settings::deflection_cutoff > 0.0 && brems_p_hard_[q].size() == n_grid) {
    // Every one of these is a straight interpolation on the index already in
    // hand. They were accessor calls, each searching the energy grid again,
    // so the lookup searched a grid of several hundred points a dozen times
    // over for one energy.
    auto on_grid = [i_grid, f](const tensor::Tensor<double>& v) {
      return v(i_grid) + f * (v(i_grid + 1) - v(i_grid));
    };

    xs.hard_elastic = xs.elastic * on_grid(elastic_p_hard_[q]);

    // Elastic only. The grouped inelastic collisions deflect as well, and by
    // no small amount, but what they deflect by is the material's, so Material
    // adds that part from the oscillator model.
    xs.soft_xs1 = xs.elastic * on_grid(elastic_mu1_soft_[q]);
    xs.soft_xs2 = xs.elastic * on_grid(elastic_mu2_soft_[q]);

    xs.hard_bremsstrahlung = xs.bremsstrahlung * on_grid(brems_p_hard_[q]);

    // The energy the grouped bremsstrahlung carries off, and its second
    // moment. The grouped inelastic collisions take energy too, but what they
    // take is the material's, so Material adds that part from the oscillator
    // model.
    xs.soft_stopping = on_grid(brems_soft_s_[q]);
    xs.soft_straggling = on_grid(brems_soft_w2_[q]);

    // In-flight annihilation is never grouped. Its photons carry away more
    // than the positron's kinetic energy, so no bound on the energy the
    // projectile gives up bounds what they can reach, and a grouped
    // annihilation would be an annihilation that did not happen.
    xs.hard_total = xs.hard_elastic + xs.annihilation + xs.hard_bremsstrahlung;
    // The larger of the two bracketing bounds, not the interpolation between
    // them. hard_majorant_ bounds the hard cross section over the energies a
    // step can reach FROM each grid point, and a linear blend of two such
    // bounds can dip below the true maximum wherever the cross section is
    // concave across the interval -- which would bias the flight silently,
    // the violation warning firing once and never again. Taking the larger is
    // rigorous and is one comparison cheaper than the blend.
    const auto& hm = hard_majorant_[q];
    xs.hard_majorant = std::max(
      xs.hard_total, std::max(hm(xs.index_grid), hm(xs.index_grid + 1)));
    xs.soft_rate = std::max(0.0, xs.total - xs.hard_total);
  } else {
    xs.hard_elastic = xs.elastic;
    xs.hard_bremsstrahlung = xs.bremsstrahlung;
    xs.hard_total = xs.total;
    xs.hard_majorant = xs.total;
    xs.soft_rate = 0.0;
    xs.soft_stopping = 0.0;
    xs.soft_straggling = 0.0;
    xs.soft_xs1 = 0.0;
    xs.soft_xs2 = 0.0;
  }

  xs.last_E = p.E();
  xs.last_q = q;
}

double Element::elastic_scatter(int q_index, double E, uint64_t* seed) const
{
  return elastic_angle_[q_index].sample(E, seed);
}

//! Elastic cross section on the electron grid, clamped as the others are
double Element::elastic_xs(int q_index, double E) const
{
  int n = electron_energy_.size();
  if (n < 2 || elastic_[q_index].size() != n)
    return 0.0;
  double f;
  int i = grid_index(electron_energy_, E, f);
  const auto& v = elastic_[q_index];
  return std::max(0.0, v(i) + f * (v(i + 1) - v(i)));
}

double Element::elastic_scatter_hard(int q_index, double E, double xs_elastic,
  double xs_hard, uint64_t* seed) const
{
  // The hard deflections are the large ones, which sit at the bottom of the
  // cumulative distribution since mu runs from backward. So the quantile is
  // restricted to the fraction of the cross section that stayed hard, and the
  // draw is exact rather than a rejection -- which matters, that fraction
  // being one part in tens of thousands.
  //
  // Both cross sections come from the caller because calculate_electron_xs has
  // already interpolated them for this energy. Asking the element for them
  // again searched the energy grid twice more per hard elastic collision, and
  // hard elastic is the commonest hard channel there is.
  //
  // The cut is exact in probability and interpolated in angle. Restricting the
  // quantile to p_hard makes the sampled fraction equal the share of the cross
  // section that stayed hard, whatever the tables do in between; what is
  // approximate is the mu the cut lands on, the distribution at E being the
  // blend of the two tables bracketing it rather than a table of its own. That
  // is the interpolation error of the angular data and not an error of the
  // split.
  double p_hard = (xs_elastic > 0.0)
                    ? std::min(1.0, std::max(0.0, xs_hard) / xs_elastic)
                    : 1.0;
  return elastic_angle_[q_index].sample_restricted(E, p_hard, seed);
}
void Element::emit_knock_on(Particle& p, double W, double e_b, double Q) const
{
  constexpr double two_m = 2.0 * MASS_ELECTRON_EV;
  double E = p.E();
  double E_knock = W - e_b;
  double phi = uniform_distribution(0., 2.0 * PI, p.current_seed());

  // Momenta of the projectile before and after, and of the transfer. The
  // smallest momentum the collision can hand over is what is left when the
  // projectile is not deflected at all, and every angle below is measured from
  // there, which keeps the cancellation out of the soft collisions that
  // dominate the count.
  double pc = std::sqrt(E * (E + two_m));
  double pc_out = std::sqrt((E - W) * (E - W + two_m));
  double cq_sq = Q * (Q + two_m);
  // (pc)^2 - (pc')^2 = W(2E - W + 2mc^2) exactly, so dividing by the sum
  // avoids subtracting two nearly equal square roots. The soft collisions that
  // dominate the count are precisely where that subtraction loses its digits,
  // and the transverse channel needs this to come back exactly forward.
  double cq_min = W * (2.0 * E - W + two_m) / (pc + pc_out);

  // The projectile is deflected through the momentum transfer. Note the
  // deflection is set by the energy the projectile actually gave up, not by
  // the kinetic energy the knock-on carries away: the two differ by the
  // binding energy, which the atom absorbs.
  double mu = 1.0 - (cq_sq - cq_min * cq_min) / (2.0 * pc * pc_out);
  p.mu() = std::max(-1.0, std::min(1.0, mu));

  // The knock-on leaves along the momentum transfer. For a free collision this
  // is the Moller partner of the angle above; for a distant one it is much
  // closer to the forward direction, since little momentum changed hands.
  double cq = std::sqrt(cq_sq);
  double mu_knock = 1.0 - (cq - cq_min) * (pc + pc_out - cq) / (2.0 * pc * cq);
  mu_knock = std::max(-1.0, std::min(1.0, mu_knock));

  Direction u_knock = rotate_angle(p.u(), mu_knock, &phi, p.current_seed());
  p.create_secondary(p.wgt(), u_knock, E_knock, ParticleType::electron());

  phi += PI;
  p.u() = rotate_angle(p.u(), p.mu(), &phi, p.current_seed());
  p.E() = E - W;
}
double Element::subshell_ionization_xs(int i_shell, double E) const
{
  int n = electron_energy_.size();
  if (n < 2 || i_shell < 0 || i_shell >= electroionization_.shape(0))
    return 0.0;
  double f;
  int i = grid_index(electron_energy_, E, f);
  double a = electroionization_(i_shell, i);
  double b = electroionization_(i_shell, i + 1);
  return std::max(0.0, a + f * (b - a));
}
namespace {

//! Polar angle of a bremsstrahlung photon, from Koch and Motz formula 2BS
//
//! EPICS carries no angular information for this channel -- the evaluation
//! states outright that the direction is left to the transport code -- so an
//! analytic form is the only option. This is the screened Schiff/Bethe-Heitler
//! shape, sampled by rejection in the reduced angle y = gamma*theta following
//! the formulation in EGSnrc, which uses it as its default for all energies.
//!
//! The emitting electron's direction is deliberately left alone. PENELOPE,
//! EGSnrc and MCNP all do the same, on the grounds that elastic scattering
//! governs the electron's deflection and the radiative recoil is negligible
//! beside it.
//!
//! \param Z Atomic number of the target
//! \param E Kinetic energy of the electron before emission [eV]
//! \param E_photon Energy of the emitted photon [eV]
//! \param seed Pseudorandom number seed pointer
//! \return Cosine of the angle between the photon and the electron
double bremsstrahlung_cos_theta(
  int Z, double E, double E_photon, uint64_t* seed)
{
  // (Zeff^(1/3)/111)^2 with Zeff^2 = Z(Z+1), the (Z+1) carrying the
  // electron-electron contribution, so the factor is (Z(Z+1))^(1/3)/111^2.
  // 1/111^2 = 8.116224e-5.
  double z_screen = 8.116224e-5 * std::cbrt(static_cast<double>(Z) * (Z + 1));
  double log_z_screen = -std::log(z_screen);

  double e_total = E + MASS_ELECTRON_EV;
  double e_final = e_total - E_photon;
  if (E_photon <= 0.0 || e_final <= MASS_ELECTRON_EV)
    return 1.0;

  double gamma = e_total / MASS_ELECTRON_EV;
  double beta = std::sqrt((gamma - 1.0) * (gamma + 1.0)) / gamma;
  double y2_max = 2.0 * beta * (1.0 + beta) * gamma * gamma;
  if (y2_max <= 0.0)
    return 1.0;
  double y2_max_inv = 1.0 / y2_max;
  double z2_max_sqrt = std::sqrt(y2_max + 1.0);

  double ratio = (e_final / MASS_ELECTRON_EV) / gamma;
  double arg1 = 1.0 + ratio * ratio;
  double arg2 = arg1 + 2.0 * ratio;

  // (2*E_final*gamma/k)^2, the argument of the screening logarithm
  double aux = 2.0 * e_final * gamma / E_photon;
  aux *= aux;
  double aux1 = aux * z_screen;
  double arg3 = (aux1 > 10.0) ? log_z_screen + (1.0 - aux1) / (aux1 * aux1)
                              : std::log(aux / (1.0 + aux1));

  double rej_max = arg1 * arg3 - arg2;
  if (!(rej_max > 0.0))
    return 1.0;

  // Rejection loop. The trial variable is drawn from the leading term and
  // rejected against the screened shape; it converges in a few iterations, but
  // cap it so that a pathological energy cannot hang a history.
  double y2 = 0.0;
  bool accepted = false;
  for (int iter = 0; iter < 1000; ++iter) {
    double xi = prn(seed);
    double test = prn(seed);
    double aux3 = z2_max_sqrt / (xi + (1.0 - xi) * z2_max_sqrt);
    test *= aux3 * rej_max;
    y2 = aux3 * aux3 - 1.0;
    double aux3_4 = aux3 * aux3 * aux3 * aux3;
    double y2s = ratio * y2 / aux3_4;
    double aux4 = 16.0 * y2s - arg2;
    double aux5 = arg1 - 4.0 * y2s;
    if (test < aux4 + aux5 * arg3) {
      accepted = true;
      break;
    }
    double aux2 = std::log(aux / (1.0 + aux1 / aux3_4));
    if (test < aux4 + aux5 * aux2) {
      accepted = true;
      break;
    }
  }

  // Falling out of the loop leaves y2 holding a trial that was rejected, which
  // would be worse than no sample at all. Emit forward instead, as the
  // degenerate cases above already do.
  if (!accepted)
    return 1.0;

  return std::max(-1.0, std::min(1.0, 1.0 - 2.0 * y2 * y2_max_inv));
}

} // namespace

double Element::sample_bremsstrahlung_energy(
  double E, uint64_t* seed, double k_min_hard) const
{
  double k_min = std::max(bremsstrahlung_photon_cutoff_, k_min_hard);
  if (E <= k_min)
    return 0.0;

  const auto& T = data::brems_e_grid;
  const auto& kappa = data::brems_k_grid;
  int n_e = T.size();
  int n_k = kappa.size();

  // Bracket the incident energy. The scaled cross section was splined onto
  // this grid against the logarithm of the energy and is linear in chi, so it
  // is interpolated the same way here. Outside the grid the nearest row is
  // used, which is what the cross section does as well.
  int i;
  double f;
  if (E <= T(0)) {
    i = 0;
    f = 0.0;
  } else if (E >= T(n_e - 1)) {
    i = n_e - 2;
    f = 1.0;
  } else {
    i = lower_bound_index(T.cbegin(), T.cend(), E);
    f = std::log(E / T(i)) / std::log(T(i + 1) / T(i));
  }

  // dsigma/dk is proportional to chi(kappa)/k, so k is sampled from 1/k over
  // [k_min, E] and the shape of chi is taken by rejection. chi is bounded by
  // its largest tabulated value on the two rows: interpolating linearly in
  // kappa and then in chi cannot leave that range.
  double chi_max = 0.0;
  for (int j = 0; j < n_k; ++j) {
    chi_max = std::max(chi_max, std::max(dcs_(i, j), dcs_(i + 1, j)));
  }
  if (chi_max <= 0.0)
    return 0.0;

  double ratio = E / k_min;
  for (int it = 0; it < MAX_REJECTION; ++it) {
    double k = k_min * std::pow(ratio, prn(seed));
    double x = k / E;
    int j = lower_bound_index(kappa.cbegin(), kappa.cend(), x);
    j = std::min(j, n_k - 2);
    double g = (x - kappa(j)) / (kappa(j + 1) - kappa(j));
    double chi_lo = dcs_(i, j) + g * (dcs_(i, j + 1) - dcs_(i, j));
    double chi_hi = dcs_(i + 1, j) + g * (dcs_(i + 1, j + 1) - dcs_(i + 1, j));
    double chi = chi_lo + f * (chi_hi - chi_lo);
    if (prn(seed) * chi_max < chi)
      return k;
  }

  // Unreachable while chi_max bounds the table it was built from. Falling back
  // to the softest photon the channel can emit keeps the emission rate right
  // and costs the spectrum nothing measurable if it ever is reached.
  return k_min;
}

double Element::annihilation_xs(double E) const
{
  // pi r_e^2 in barns, with the classical electron radius written as
  // alpha^2 a_0 so that it follows from the constants already tabulated
  constexpr double BOHR_RADIUS_CM =
    PLANCK_C * FINE_STRUCTURE / (2.0 * PI * MASS_ELECTRON_EV) * 1.0e-8;
  constexpr double R_E = BOHR_RADIUS_CM / (FINE_STRUCTURE * FINE_STRUCTURE);
  constexpr double PI_R_E_SQ = PI * R_E * R_E * 1.0e24;

  double gamma = 1.0 + E / MASS_ELECTRON_EV;
  double g_sq = gamma * gamma;
  if (g_sq <= 1.0 + 1.0e-12)
    return 0.0;
  double s = std::sqrt(g_sq - 1.0);

  // Heitler, per electron. It grows as 1/beta as the positron slows, which is
  // why a positron that reaches the cutoff annihilates rather than lingers.
  double sigma =
    PI_R_E_SQ / (gamma + 1.0) *
    ((g_sq + 4.0 * gamma + 1.0) / (g_sq - 1.0) * std::log(gamma + s) -
      (gamma + 3.0) / s);
  return Z_ * std::max(sigma, 0.0);
}

void Element::annihilation(Particle& p) const
{
  // Energy available to the two photons: the positron's kinetic energy and
  // both rest masses, the electron being taken as free and at rest.
  double avail = p.E() + 2.0 * MASS_ELECTRON_EV;
  double a = avail / MASS_ELECTRON_EV; // gamma + 1
  double t = a - 2.0;                  // gamma - 1
  double pc = std::sqrt(a * t);        // sqrt(gamma^2 - 1)
  double pot = pc / t;

  // Sample the fraction of the available energy taken by the first photon.
  // It runs between ep0 and 1 - ep0, and is drawn from 1/ep with the rest of
  // Heitler's spectrum taken by rejection. The rejection function is written
  // normalised to its own maximum, as in EGSnrc.
  double ep0 = 1.0 / (a + pc);
  double span = std::log((1.0 - ep0) / ep0);
  double ep = ep0;
  for (int it = 0; it < MAX_REJECTION; ++it) {
    ep = ep0 * std::exp(span * prn(p.current_seed()));
    double arg = ep * a - 1.0;
    double rejection = 1.0 - arg * arg / (ep * (a * a - 2.0));
    if (prn(p.current_seed()) <= rejection)
      break;
  }

  double E_1 = avail * ep;
  double E_2 = avail - E_1;

  // Both polar angles follow from the photon energies; the two are coplanar
  // with the incident direction and on opposite sides of it.
  double phi = uniform_distribution(0., 2.0 * PI, p.current_seed());
  double mu_1 =
    std::max(-1.0, std::min(1.0, (E_1 - MASS_ELECTRON_EV) * pot / E_1));
  double mu_2 =
    std::max(-1.0, std::min(1.0, (E_2 - MASS_ELECTRON_EV) * pot / E_2));
  Direction u_1 = rotate_angle(p.u(), mu_1, &phi, p.current_seed());
  double phi_2 = phi + PI;
  Direction u_2 = rotate_angle(p.u(), mu_2, &phi_2, p.current_seed());
  p.create_secondary(p.wgt(), u_1, E_1, ParticleType::photon());
  p.create_secondary(p.wgt(), u_2, E_2, ParticleType::photon());

  // The positron is gone. Zeroing the weight as well tells the post-collision
  // cutoff handling that this particle has already annihilated, and the
  // POSITRON_ANNIHILATION marker gives the heating score the Q value that
  // balances the two rest masses the photons carry away.
  p.E() = 0.0;
  p.wgt() = 0.0;
  p.event() = TallyEvent::ABSORB;
  p.event_mt() = POSITRON_ANNIHILATION;
}

void Element::bremsstrahlung(Particle& p, double k_min) const
{
  int n_split = settings::bremsstrahlung_split;

  // Unsplit, and the common case: one emission, the electron loses it.
  if (n_split == 1) {
    double E_photon =
      this->sample_bremsstrahlung_energy(p.E(), p.current_seed(), k_min);
    if (E_photon <= 0.0)
      return;
    double mu = bremsstrahlung_cos_theta(Z_, p.E(), E_photon, p.current_seed());
    Direction u = rotate_angle(p.u(), mu, nullptr, p.current_seed());
    p.E() -= E_photon;
    p.create_secondary(p.wgt(), u, E_photon, ParticleType::photon());
    return;
  }

  // Split. The emissions are drawn independently rather than copied, because
  // what makes a photon worth having here is its energy: copying one photon
  // n times gives n tries at the same energy, while drawing n times gives n
  // tries at reaching the thin high-energy end of the spectrum, which is the
  // part an answer sensitive to the shape of the photon field, rather than to
  // how much energy it carries, is starved of.
  //
  // The electron cannot lose all of them -- it emitted one photon, not n. It
  // loses the first draw, which is an unbiased sample of what one emission
  // takes, so the electron's history stays a fair one and the photon field is
  // right in expectation. Energy is conserved in the mean rather than event by
  // event, which is what splitting costs and why this is off by default.
  double w = p.wgt() / n_split;
  double E_first = -1.0;
  double banked = 0.0;
  for (int i = 0; i < n_split; ++i) {
    // Every draw is made at the energy the electron came in with, so the loop
    // must not touch p.E() until it is done
    double E_photon =
      this->sample_bremsstrahlung_energy(p.E(), p.current_seed(), k_min);
    if (E_photon <= 0.0)
      continue;
    if (E_first < 0.0)
      E_first = E_photon;

    // Below the transport cutoff the photon is discarded by create_secondary
    // anyway, so there is nothing to split and the angle is not worth drawing
    int i_photon = ParticleType::photon().transport_index();
    if (E_photon < settings::energy_cutoff[i_photon])
      continue;

    double mu = bremsstrahlung_cos_theta(Z_, p.E(), E_photon, p.current_seed());
    Direction u = rotate_angle(p.u(), mu, nullptr, p.current_seed());
    if (p.create_secondary(w, u, E_photon, ParticleType::photon()))
      banked += E_photon;
  }
  if (E_first < 0.0)
    return;

  p.E() -= E_first;

  // create_secondary() banked each photon's whole energy, but each carries
  // only 1/n of the weight. Correct the balance the heating score is built on.
  p.bank_second_E() += (1.0 / n_split - 1.0) * banked;
}

} // namespace openmc
