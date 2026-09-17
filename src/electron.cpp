#include "openmc/photon.h"

#include "openmc/array.h"
#include "openmc/bremsstrahlung.h"
#include "openmc/constants.h"
#include "openmc/distribution_multi.h"
#include "openmc/hdf5_interface.h"
#include "openmc/material.h"
#include "openmc/math_functions.h"
#include "openmc/message_passing.h"
#include "openmc/nuclide.h"
#include "openmc/particle.h"
#include "openmc/physics.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/settings.h"

#include "openmc/tensor.h"

#include <algorithm> // for max, min
#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <stdexcept>
#include <tuple> // for tie

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
  int n = grid.size();
  auto out = tensor::zeros<double>({n});
  if (energy.size() < 2)
    return out;

  for (int i = 0; i < n; ++i) {
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

    // Transport moments of the elastic distribution, tabulated once here onto
    // the electron energy grid. <1-mu> is what sets the scale a condensed
    // history step may cover: after a path s the mean deflection is
    // exp(-s n sigma_el <1-mu>), so 1/(n sigma_el <1-mu>) is the first
    // transport mean free path. Computing them at load costs one pass over
    // data already in memory and needs nothing the library does not ship.
    vector<double> m_energy, m1, m2;
    elastic_angle_[q].transport_moments(m_energy, m1, m2);
    elastic_mu1_[q] = moments_on_grid(electron_energy_, m_energy, m1);
    elastic_mu2_[q] = moments_on_grid(electron_energy_, m_energy, m2);

    // Where the same distribution splits into soft and hard, for a mixed
    // condensed-history step. C1 = 0 is the default and means every collision
    // is hard, so this costs nothing until a run asks for it.
    if (settings::electron_max_step_deflection > 0.0) {
      vector<double> s_energy, mu_cut, p_hard, m1_soft, m2_soft;
      elastic_angle_[q].restricted_moments(
        settings::electron_max_step_deflection, s_energy, mu_cut, p_hard,
        m1_soft, m2_soft);
      vector<double> dcut(mu_cut.size());
      for (int i = 0; i < mu_cut.size(); ++i) {
        dcut[i] = std::max(0.0, 1.0 - mu_cut[i]);
      }
      elastic_dcut_[q] = moments_on_grid(electron_energy_, s_energy, dcut);
      elastic_p_hard_[q] = moments_on_grid(electron_energy_, s_energy, p_hard);
      elastic_mu1_soft_[q] =
        moments_on_grid(electron_energy_, s_energy, m1_soft);
      elastic_mu2_soft_[q] =
        moments_on_grid(electron_energy_, s_energy, m2_soft);
    }
  }
  close_group(rgroup);

  // Read excitation
  rgroup = open_group(group, "excitation");
  read_dataset(rgroup, "xs", excitation_);
  hid_t dset = open_dataset(rgroup, "energy_loss");
  excitation_energy_loss_ = Tabulated1D {dset};
  close_dataset(dset);
  close_group(rgroup);

  // Read ionization
  rgroup = open_group(group, "ionization");
  read_dataset(rgroup, "xs", electroionization_);
  vector<std::string> designators;
  read_attribute(rgroup, "designators", designators);
  for (auto designator : designators) {
    hid_t shell_group = open_group(rgroup, designator.c_str());
    // Knock-on spectra are anchored at the subshell binding energy and end at
    // the kinematic limit, so they are neither self-similar nor safe to remap
    // onto an interpolated endpoint range; see ElectroionizationSpectrum.
    ionization_dist_.push_back(
      make_unique<ElectroionizationSpectrum>(shell_group));
    close_group(shell_group);
  }
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

  this->compute_moller_majorant();
  this->compute_bhabha_xs();

  // The soft/hard split of the inelastic channels, which needs every one of
  // them loaded and so comes last. Skipped unless a run asks for condensed
  // history, since it is the only thing that reads it.
  if (settings::electron_max_step_deflection > 0.0) {
    this->compute_soft_inelastic();
  }
}

namespace {

//! The free Moller and Bhabha differential cross sections, as functions of the
//! fraction eps = W/T of its kinetic energy the projectile transfers
//
//! Both carry the same leading constant, so only their shapes are written here
//! and the constant cancels wherever the two are divided. The coefficients are
//! PENELOPE's.
struct FreeCollision {
  double b1, b2, b3, b4; //!< Bhabha
  double amol, moller_c; //!< Moller

  explicit FreeCollision(double E)
  {
    double gamma = 1.0 + E / MASS_ELECTRON_EV;
    // ((gamma-1)/gamma)^2, the constant term of the Moller shape and the
    // common factor of every Bhabha coefficient
    amol = std::pow((gamma - 1.0) / gamma, 2);
    double g12 = (gamma + 1.0) * (gamma + 1.0);
    b1 = amol * (2.0 * g12 - 1.0) / (gamma * gamma - 1.0);
    b2 = amol * (3.0 + 1.0 / g12);
    b3 = amol * 2.0 * gamma * (gamma - 1.0) / g12;
    b4 = amol * (gamma - 1.0) * (gamma - 1.0) / g12;
    moller_c = (2.0 * gamma - 1.0) / (gamma * gamma);
  }

  double bhabha(double x) const
  {
    return (1.0 + x * (-b1 + x * (b2 + x * (-b3 + x * b4)))) / (x * x);
  }

  double moller(double x) const
  {
    double u = 1.0 - x;
    return 1.0 / (x * x) + 1.0 / (u * u) + amol - moller_c / (x * u);
  }

  //! Ratio of the two. It tends to 1 as x tends to 0, where both become
  //! Rutherford's 1/x^2, so it leaves the soft collisions -- which are almost
  //! all of them -- alone. Relativistically it is 1 - 2x to good accuracy.
  double ratio(double x) const { return this->bhabha(x) / this->moller(x); }
};

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
double bhabha_integral(
  const FreeCollision& c, double E, double W_lo, double W_hi)
{
  return bhabha_moment(c, E, W_lo, W_hi, 0);
}

} // namespace

namespace detail {

double bhabha_moment(double E, double W_lo, double W_hi, int order)
{
  return openmc::bhabha_moment(FreeCollision {E}, E, W_lo, W_hi, order);
}

} // namespace detail

namespace {

//! 2 pi r_e^2 m_e c^2 in [b eV], the constant both cross sections carry. The
//! classical electron radius is written as alpha^2 a_0 so that it follows from
//! the constants already tabulated rather than adding one.
constexpr double BOHR_RADIUS_CM =
  PLANCK_C * FINE_STRUCTURE / (2.0 * PI * MASS_ELECTRON_EV) * 1.0e-8;
constexpr double R_E = BOHR_RADIUS_CM / (FINE_STRUCTURE * FINE_STRUCTURE);
constexpr double COLLISION_CONST =
  2.0 * PI * 1.0e24 * R_E * R_E * MASS_ELECTRON_EV;

} // namespace

void Element::compute_moller_majorant()
{
  int n_energy = electron_energy_.size();
  moller_majorant_ = tensor::Tensor<double>({static_cast<size_t>(n_energy)});

  // The evaluated knock-on spectra describe a Moller collision. A positron's
  // spectrum is the same thing reweighted by the ratio of the two free cross
  // sections: the final state is identical, an electron ejected with W - B, so
  // whatever the binding does to one it does to the other and it cancels in
  // the ratio. That is a far weaker assumption than either cross section being
  // free, and it is the one PENELOPE makes when it applies Moller and Bhabha
  // per oscillator.
  //
  // The reweighting is done by rejection during the collision, so the cross
  // section has to be a majorant of the true one: the largest the ratio gets
  // over the accessible transfers. Above about 1 MeV that is 1, the ratio
  // being 1 - 2x; below it the ratio rises above 1 near x = 0.3 and the
  // majorant follows it.
  constexpr int N_SCAN = 512;
  for (int j = 0; j < n_energy; ++j) {
    FreeCollision c {electron_energy_(j)};
    double peak = 1.0;
    for (int k = 0; k <= N_SCAN; ++k) {
      // Logarithmic in x, since the ratio varies fastest near the ends
      double x = std::exp(
        std::log(1.0e-8) + k * (std::log(0.5) - std::log(1.0e-8)) / N_SCAN);
      peak = std::max(peak, c.ratio(x));
    }
    // A majorant that is a shade too small would bias the sampling, so the
    // scan's own resolution is paid for here
    moller_majorant_(j) = peak * 1.001;
  }
}

void Element::compute_bhabha_xs()
{
  int n_shell = electroionization_.shape(0);
  int n_energy = electron_energy_.size();
  bhabha_ = tensor::Tensor<double>(
    {static_cast<size_t>(n_shell), static_cast<size_t>(n_energy)});

  for (int i = 0; i < n_shell; ++i) {
    const auto& shell {shells_[electron_shell_map_[i]]};
    double B = shell.binding_energy;
    double n_e = shell.num_electrons;

    for (int j = 0; j < n_energy; ++j) {
      double E = electron_energy_(j);
      bhabha_(i, j) = 0.0;
      if (n_e <= 0.0 || E <= B)
        continue;

      // A Moller collision gives the knock-on at most half of what is left
      // after the binding energy is paid, so the largest transfer it can make
      // is (T + B)/2. A positron may transfer everything. The gap between the
      // two is this channel, and it lies far above every binding energy.
      double W_lo = 0.5 * (E + B);
      FreeCollision c {E};
      double gamma = 1.0 + E / MASS_ELECTRON_EV;
      double beta_sq = 1.0 - 1.0 / (gamma * gamma);
      bhabha_(i, j) =
        n_e * COLLISION_CONST / beta_sq * bhabha_integral(c, E, W_lo, E);
    }
  }
}

namespace {

//! Transport cutoff of the projectile itself
double own_cutoff(int q_index)
{
  int index = (q_index == 0) ? ParticleType::electron().transport_index()
                             : ParticleType::positron().transport_index();
  return settings::energy_cutoff[index];
}

} // namespace

double soft_projectile_headroom(int q_index, double E)
{
  return std::max(0.0, E - own_cutoff(q_index));
}

namespace {

//! Largest share of a step's energy budget one grouped collision may carry
//!
//! The step describes that energy by two moments, and the transfers are
//! distributed as 1/W^2, so the variance sits in the few largest of them. A
//! tenth leaves about ten of them to share it, which is the fewest that makes
//! a mean and a variance mean anything.
constexpr double MAX_SOFT_LOSS_SHARE = 0.1;

//! Energy a step is allowed to lose to the grouped collisions
double soft_loss_budget(int q_index, double E)
{
  return std::min(settings::electron_max_step_energy_loss * E,
    soft_projectile_headroom(q_index, E));
}

} // namespace

double soft_collision_cutoff(int q_index, double E)
{
  double photon =
    settings::energy_cutoff[ParticleType::photon().transport_index()];
  double electron =
    settings::energy_cutoff[ParticleType::electron().transport_index()];
  return std::max(
    0.0, std::min(std::min(photon, electron),
           std::min(soft_projectile_headroom(q_index, E),
             MAX_SOFT_LOSS_SHARE * soft_loss_budget(q_index, E))));
}

double soft_radiative_cutoff(int q_index, double E)
{
  double photon =
    settings::energy_cutoff[ParticleType::photon().transport_index()];
  return std::max(0.0,
    std::min(photon, std::min(soft_projectile_headroom(q_index, E),
                       MAX_SOFT_LOSS_SHARE * soft_loss_budget(q_index, E))));
}

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

void Element::compute_soft_inelastic()
{
  int n_energy = electron_energy_.size();
  int n_shell = electroionization_.shape(0);
  auto shape_1d = std::vector<size_t> {static_cast<size_t>(n_energy)};
  auto shape_2d = std::vector<size_t> {
    static_cast<size_t>(n_shell), static_cast<size_t>(n_energy)};

  for (int q = 0; q < 2; ++q) {
    inelastic_soft_s_[q] = tensor::zeros<double>(shape_1d);
    inelastic_soft_w2_[q] = tensor::zeros<double>(shape_1d);
    excitation_p_hard_[q] = tensor::zeros<double>(shape_1d);
    brems_p_hard_[q] = tensor::zeros<double>(shape_1d);
    ionization_p_hard_[q] = tensor::zeros<double>(shape_2d);
    ionization_hard_xs_[q] = tensor::zeros<double>(shape_1d);
  }
  bhabha_p_hard_ = tensor::zeros<double>(shape_2d);
  bhabha_hard_xs_ = tensor::zeros<double>(shape_1d);

  const auto& T = data::brems_e_grid;
  int n_brems_e = T.size();
  int n_kappa = data::brems_k_grid.size();
  vector<double> chi(n_kappa);

  for (int j = 0; j < n_energy; ++j) {
    double E = electron_energy_(j);
    FreeCollision c {E};
    double gamma = 1.0 + E / MASS_ELECTRON_EV;
    double beta_sq = 1.0 - 1.0 / (gamma * gamma);

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

    // Everything below is per projectile charge, because the two thresholds
    // differ: a positron's own cutoff bounds how much a grouped event may take
    // from it, and that cutoff is not the electron's.
    for (int q = 0; q < 2; ++q) {
      double w_cc = soft_collision_cutoff(q, E);
      double w_cr = soft_radiative_cutoff(q, E);
      double headroom = soft_projectile_headroom(q, E);
      double s = 0.0;
      double w2 = 0.0;

      // Atomic excitation, which emits nothing at all -- OpenMC deposits the
      // de-excitation energy where the collision happened -- so the only bound
      // on it is that it leave the projectile above its cutoff. The evaluated
      // loss is one number per collision rather than a distribution, so its
      // second moment is the square of that loss.
      double loss = std::max(0.0, E - this->excitation(E));
      if (loss > 0.0 && loss < headroom) {
        s += excitation_(j) * loss;
        w2 += excitation_(j) * loss * loss;
        // Note an excitation collision changes the projectile's energy and
        // not its direction, so however much of the stopping power it carries
        // it contributes nothing to the transport cross section computed in
        // compute_inelastic_transport().
      } else {
        excitation_p_hard_[q](j) = 1.0;
      }

      // Electroionization, subshell by subshell. The cut is on the energy the
      // projectile gives up, so it sits at W_cc - B in the knock-on spectrum:
      // the atom keeps the binding energy and only the rest is carried away.
      for (int i = 0; i < n_shell; ++i) {
        const auto& shell {shells_[electron_shell_map_[i]]};
        double B = shell.binding_energy;
        double sigma = electroionization_(i, j);
        double e_cut = w_cc - B;

        double xi_cut = 0.0;
        if (sigma > 0.0 && e_cut > 0.0) {
          // A positron's spectrum is the tabulated one reweighted by the free
          // Bhabha-to-Moller ratio. The weighting goes inside the integral
          // rather than onto the result: it varies across the spectrum, and it
          // is exactly what the rejection in ionization() applies collision by
          // collision.
          std::function<double(double)> weight = [&c, B, E](double e_out) {
            return c.ratio((e_out + B) / E);
          };
          double m0, m1, m2;
          ionization_dist_[i]->restricted_moments(
            E, e_cut, q == 0 ? nullptr : &weight, xi_cut, m0, m1, m2);
          s += sigma * (m1 + B * m0);
          w2 += sigma * (m2 + 2.0 * B * m1 + B * B * m0);
        }
        ionization_p_hard_[q](i, j) = 1.0 - xi_cut;
        ionization_hard_xs_[q](j) += sigma * (1.0 - xi_cut);

        // Transfers above the Moller limit, which only a positron can make.
        // They lie far above every binding energy, so this channel is usually
        // hard in its entirety -- but not for a projectile barely above the
        // cutoff, where half its energy is still below W_cc.
        double n_e = shell.num_electrons;
        double W_lo = 0.5 * (E + B);
        if (q == 1 && n_e > 0.0 && E > B && W_lo < E) {
          double K = n_e * COLLISION_CONST / beta_sq;
          double W_soft = std::min(w_cc, E);
          if (W_soft > W_lo) {
            s += K * bhabha_moment(c, E, W_lo, W_soft, 1);
            w2 += K * bhabha_moment(c, E, W_lo, W_soft, 2);
          }
          double total = bhabha_moment(c, E, W_lo, E, 0);
          double hard = bhabha_moment(c, E, std::max(W_lo, W_soft), E, 0);
          bhabha_p_hard_(i, j) = (total > 0.0) ? hard / total : 0.0;
          bhabha_hard_xs_(j) += bhabha_(i, j) * bhabha_p_hard_(i, j);
        }
      }

      // Bremsstrahlung. The rate is taken from the library rather than rebuilt
      // here, and only the fraction of it below the cutoff is computed from
      // the scaled cross section, so the total emission rate stays exactly
      // what the single-event transport uses.
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
        s += xs * E * chi_integral(chi, kappa_min, kappa_cut, 1) / total;
        w2 += xs * E * E * chi_integral(chi, kappa_min, kappa_cut, 2) / total;
      } else {
        brems_p_hard_[q](j) = 1.0;
      }

      inelastic_soft_s_[q](j) = s;
      inelastic_soft_w2_[q](j) = w2;
    }
  }
}

int Element::sample_bhabha_shell(Particle& p, bool hard) const
{
  const auto& xs {p.electron_xs(index_)};
  int n_shell = bhabha_.shape(0);
  int i_grid = xs.index_grid;
  double f = xs.interp_factor;

  auto weight = [&](int i) {
    double sigma =
      bhabha_(i, i_grid) + f * (bhabha_(i, i_grid + 1) - bhabha_(i, i_grid));
    return hard ? sigma * this->bhabha_hard_fraction(i, p.E()) : sigma;
  };

  double total = 0.0;
  for (int i = 0; i < n_shell; ++i) {
    total += weight(i);
  }
  double cutoff = prn(p.current_seed()) * total;
  double prob = 0.0;
  for (int i = 0; i < n_shell; ++i) {
    prob += weight(i);
    if (prob > cutoff)
      return i;
  }
  return n_shell - 1;
}

void Element::bhabha(Particle& p, int i_shell, double w_min) const
{
  double E = p.E();
  double B = shells_[electron_shell_map_[i_shell]].binding_energy;
  // A collision ending a condensed-history step transfers more than the
  // step's cutoff; below it the channel was grouped into the stopping power
  double W_lo = std::max(0.5 * (E + B), w_min);
  if (W_lo >= E)
    return;

  // Sample the transfer from 1/W^2 over [W_lo, T] and take the rest of the
  // Bhabha shape by rejection; it is at most 1 on this interval
  FreeCollision c {E};
  double W;
  while (true) {
    double xi = prn(p.current_seed());
    W = W_lo * E / (E - xi * (E - W_lo));
    if (prn(p.current_seed()) < c.bhabha(W / E) * (W * W) / (E * E))
      break;
  }

  this->emit_knock_on(p, W, B, W);
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
  // ever. That is reachable, because the default electron cutoff is zero.
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

  // Calculate microscopic excitation cross section
  xs.excitation =
    excitation_(i_grid) + f * (excitation_(i_grid + 1) - excitation_(i_grid));

  // Calculate microscopic ionization cross section
  const auto ion_i = electroionization_.slice(tensor::all, i_grid).sum();
  const auto ion_ip1 = electroionization_.slice(tensor::all, i_grid + 1).sum();
  xs.ionization = ion_i + f * (ion_ip1 - ion_i);

  xs.bhabha = 0.0;
  if (p.type().is_positron()) {
    // A positron's spectrum is the evaluated one reweighted by the free
    // Bhabha-to-Moller ratio, applied by rejection inside the collision. That
    // makes this a majorant rather than the cross section itself: raising it
    // here and declining a fraction of the collisions there leaves the rate at
    // the reweighted integral without anyone having to evaluate that integral.
    xs.ionization *=
      moller_majorant_(i_grid) +
      f * (moller_majorant_(i_grid + 1) - moller_majorant_(i_grid));

    // Transfers above the Moller limit, which the evaluated spectra cannot
    // reach at all, are a channel of their own
    const auto bha_i = bhabha_.slice(tensor::all, i_grid).sum();
    const auto bha_ip1 = bhabha_.slice(tensor::all, i_grid + 1).sum();
    xs.bhabha = bha_i + f * (bha_ip1 - bha_i);
  }

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

  // Calculate microscopic total cross section
  xs.total = xs.elastic + xs.excitation + xs.ionization + xs.bhabha +
             xs.annihilation + xs.bremsstrahlung;

  // Split it into the part a condensed-history step transports one collision
  // at a time and the part it groups. Without condensed history every channel
  // is hard, which is what leaves the transport below untouched.
  if (settings::electron_max_step_deflection > 0.0 &&
      inelastic_soft_s_[q].size() == n_grid) {
    // Every one of these is a straight interpolation on the index already in
    // hand. They were accessor calls, each searching the energy grid again --
    // elastic_split() and inelastic_soft() among them, so the lookup searched
    // a grid of several hundred points a dozen times over for one energy.
    auto on_grid = [i_grid, f](const tensor::Tensor<double>& v) {
      return v(i_grid) + f * (v(i_grid + 1) - v(i_grid));
    };

    xs.hard_elastic = xs.elastic * on_grid(elastic_p_hard_[q]);

    // Elastic only. The grouped inelastic collisions deflect as well, and by
    // no small amount, but what they deflect by depends on the material -- see
    // compute_inelastic_transport() -- so Material adds that part.
    xs.soft_xs1 = xs.elastic * on_grid(elastic_mu1_soft_[q]);
    xs.soft_xs2 = xs.elastic * on_grid(elastic_mu2_soft_[q]);

    xs.hard_excitation = xs.excitation * on_grid(excitation_p_hard_[q]);

    // Electroionization resolves its split per subshell, so the hard part is
    // the shell cross sections weighted by their own fractions rather than the
    // total weighted by an average -- summed at load, since it depends on
    // nothing but the energy. A positron's cross section carries the majorant
    // it is sampled with, and the fractions are quantiles of the spectrum, so
    // the majorant passes straight through.
    double ion_hard = on_grid(ionization_hard_xs_[q]);
    if (p.type().is_positron()) {
      ion_hard *= on_grid(moller_majorant_);
    }
    xs.hard_ionization = std::min(xs.ionization, ion_hard);

    xs.hard_bhabha = std::min(xs.bhabha, on_grid(bhabha_hard_xs_));

    xs.hard_bremsstrahlung = xs.bremsstrahlung * on_grid(brems_p_hard_[q]);

    // In-flight annihilation is never grouped. Its photons carry away more
    // than the positron's kinetic energy, so no bound on the energy the
    // projectile gives up bounds what they can reach, and a grouped
    // annihilation would be an annihilation that did not happen.
    xs.hard_total = xs.hard_elastic + xs.hard_excitation + xs.hard_ionization +
                    xs.hard_bhabha + xs.annihilation + xs.hard_bremsstrahlung;
    xs.soft_rate = std::max(0.0, xs.total - xs.hard_total);
    xs.soft_stopping = on_grid(inelastic_soft_s_[q]);
    xs.soft_straggling = on_grid(inelastic_soft_w2_[q]);
  } else {
    xs.hard_elastic = xs.elastic;
    xs.hard_excitation = xs.excitation;
    xs.hard_ionization = xs.ionization;
    xs.hard_bhabha = xs.bhabha;
    xs.hard_bremsstrahlung = xs.bremsstrahlung;
    xs.hard_total = xs.total;
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

double Element::elastic_scatter_hard(
  int q_index, double E, uint64_t* seed) const
{
  // The hard deflections are the large ones, which sit at the bottom of the
  // cumulative distribution since mu runs from backward. So the quantile is
  // restricted to the fraction of the cross section that stayed hard, and the
  // draw is exact rather than a rejection -- which matters, that fraction
  // being one part in tens of thousands.
  auto split = this->elastic_split(q_index, E);
  double xs = std::max(0.0, split.xs_hard);
  double total = elastic_xs(q_index, E);
  double p_hard = (total > 0.0) ? std::min(1.0, xs / total) : 1.0;
  return elastic_angle_[q_index].sample_restricted(E, p_hard, seed);
}

double Element::elastic_transport_xs(int q_index, double E, int order) const
{
  const auto& moment =
    (order == 1) ? elastic_mu1_[q_index] : elastic_mu2_[q_index];
  int n = electron_energy_.size();
  if (n < 2 || moment.size() != n)
    return 0.0;

  // Same clamped lookup the cross sections use: outside the tabulated range a
  // linear extrapolation of a quantity falling as a power of the energy goes
  // negative, and a negative transport cross section is a negative step length.
  int i =
    upper_bound_index(electron_energy_.cbegin(), electron_energy_.cend(), E);
  i = std::max(0, std::min(i, n - 2));
  double f =
    (E - electron_energy_(i)) / (electron_energy_(i + 1) - electron_energy_(i));
  f = std::max(0.0, std::min(1.0, f));

  double xs = elastic_[q_index](i) +
              f * (elastic_[q_index](i + 1) - elastic_[q_index](i));
  double mu = moment(i) + f * (moment(i + 1) - moment(i));
  return std::max(0.0, xs * mu);
}

ElasticSplit Element::elastic_split(int q_index, double E) const
{
  ElasticSplit split;
  int n = electron_energy_.size();
  if (n < 2 || elastic_[q_index].size() != n)
    return split;

  // Same clamped lookup the cross sections use
  int i =
    upper_bound_index(electron_energy_.cbegin(), electron_energy_.cend(), E);
  i = std::max(0, std::min(i, n - 2));
  double f =
    (E - electron_energy_(i)) / (electron_energy_(i + 1) - electron_energy_(i));
  f = std::max(0.0, std::min(1.0, f));

  auto interp = [i, f](const tensor::Tensor<double>& v) {
    return v(i) + f * (v(i + 1) - v(i));
  };

  double xs = std::max(0.0, interp(elastic_[q_index]));

  // Single-event transport, which is what C1 = 0 asks for and what a library
  // without the split tabulated gets: the cutoff sits at mu = 1, every
  // collision is hard and nothing is grouped.
  if (elastic_p_hard_[q_index].size() != n) {
    split.xs_hard = xs;
    return split;
  }

  split.mu_cut = 1.0 - std::max(0.0, interp(elastic_dcut_[q_index]));
  split.xs_hard = std::max(0.0, xs * interp(elastic_p_hard_[q_index]));
  split.xs1_soft = std::max(0.0, xs * interp(elastic_mu1_soft_[q_index]));
  split.xs2_soft = std::max(0.0, xs * interp(elastic_mu2_soft_[q_index]));
  return split;
}

void Element::inelastic_soft(int q_index, double E, double& s, double& w2) const
{
  s = 0.0;
  w2 = 0.0;
  int n = electron_energy_.size();
  if (n < 2 || inelastic_soft_s_[q_index].size() != n)
    return;

  double f;
  int i = grid_index(electron_energy_, E, f);
  const auto& sv = inelastic_soft_s_[q_index];
  const auto& wv = inelastic_soft_w2_[q_index];
  s = std::max(0.0, sv(i) + f * (sv(i + 1) - sv(i)));
  w2 = std::max(0.0, wv(i) + f * (wv(i + 1) - wv(i)));
}

void Element::compute_inelastic_transport(int q_index,
  const vector<double>& w_r, const vector<double>& delta,
  tensor::Tensor<double>& xs1) const
{
  constexpr double two_m = 2.0 * MASS_ELECTRON_EV;
  int n_energy = electron_energy_.size();
  int n_shell = electroionization_.shape(0);
  xs1 =
    tensor::zeros<double>(std::vector<size_t> {static_cast<size_t>(n_energy)});
  if (settings::electron_max_step_deflection <= 0.0 || w_r.size() != n_shell)
    return;

  for (int j = 0; j < n_energy; ++j) {
    double E = electron_energy_(j);
    double w_cc = soft_collision_cutoff(q_index, E);
    if (!(w_cc > 0.0) || E <= 0.0)
      continue;

    double pc = std::sqrt(E * (E + two_m));
    double gamma = 1.0 + E / MASS_ELECTRON_EV;
    double beta_sq = 1.0 - 1.0 / (gamma * gamma);
    FreeCollision c {E};

    // Transverse strength of a distant collision, which is the one the
    // density effect acts on. It hands over no momentum at all, so every
    // collision that goes this way is deflected through exactly nothing.
    double c_tra =
      -std::log1p(-beta_sq) - beta_sq - (j < delta.size() ? delta[j] : 0.0);

    double total = 0.0;
    for (int i = 0; i < n_shell; ++i) {
      const auto& shell {shells_[electron_shell_map_[i]]};
      double B = shell.binding_energy;
      double sigma = electroionization_(i, j);
      double e_cut = w_cc - B;
      if (sigma <= 0.0 || e_cut <= 0.0 || w_r[i] <= 0.0)
        continue;
      double n_e = shell.num_electrons;

      // The deflection one collision makes, averaged over the recoil the
      // model would have sampled for it. Written from the same branches
      // sample_recoil() takes, so that what is removed from the discrete
      // channel is exactly what is added to the grouped one.
      auto deflection = [&](double e_out, double density) {
        double W = e_out + B;
        if (!(W > 0.0) || W >= E)
          return 0.0;
        double pc_out = std::sqrt((E - W) * (E - W + two_m));
        double cq_min = W * (2.0 * E - W + two_m) / (pc + pc_out);
        double cq_min_sq = cq_min * cq_min;
        double q_min =
          cq_min_sq /
          (std::sqrt(MASS_ELECTRON_EV * MASS_ELECTRON_EV + cq_min_sq) +
            MASS_ELECTRON_EV);
        double denom = 2.0 * pc * pc_out;
        if (!(denom > 0.0))
          return 0.0;

        // A close collision leaves the whole transfer as recoil
        double mu_close = std::max(0.0, (W * (W + two_m) - cq_min_sq) / denom);
        if (q_min >= w_r[i])
          return mu_close;

        // Share of the collisions that struck a single electron, which is
        // what the free cross section can account for
        double p_close = 1.0;
        if (density > 0.0) {
          double sigma_free =
            n_e * COLLISION_CONST / beta_sq * c.moller(W / E) / (E * E);
          p_close = std::min(1.0, sigma_free / (sigma * density));
        } else if (W > w_r[i]) {
          return mu_close;
        }

        // Distant. The longitudinal recoil runs over [q_min, w_r] as
        // 1/(Q(Q+2m)), so its mean deflection is elementary; the transverse
        // part contributes nothing.
        double c_lon =
          std::log(w_r[i] * (q_min + two_m) / (q_min * (w_r[i] + two_m)));
        if (!(c_lon > 0.0))
          return p_close * mu_close;
        double mu_lon =
          std::max(0.0, (two_m * (w_r[i] - q_min) / c_lon - cq_min_sq) / denom);
        double f_lon = (c_tra > 0.0) ? c_lon / (c_tra + c_lon) : 1.0;
        return p_close * mu_close + (1.0 - p_close) * f_lon * mu_lon;
      };

      total +=
        sigma * ionization_dist_[i]->restricted_integral(E, e_cut, deflection);
    }

    // Transfers above the Moller limit, which only a positron makes and which
    // are free collisions by construction
    if (q_index == 1) {
      for (int i = 0; i < n_shell; ++i) {
        const auto& shell {shells_[electron_shell_map_[i]]};
        double B = shell.binding_energy;
        double n_e = shell.num_electrons;
        double W_lo = 0.5 * (E + B);
        double W_soft = std::min(w_cc, E);
        if (n_e <= 0.0 || E <= B || W_soft <= W_lo)
          continue;
        // First order in W is enough here: these transfers are soft only for
        // a projectile barely above its cutoff, where the channel is small
        double K = n_e * COLLISION_CONST / beta_sq;
        total += K * bhabha_moment(c, E, W_lo, W_soft, 1) * MASS_ELECTRON_EV /
                 (E * (E + two_m));
      }
    }

    xs1(j) = std::max(0.0, total);
  }
}

double Element::excitation_hard_fraction(int q_index, double E) const
{
  int n = electron_energy_.size();
  if (n < 2 || excitation_p_hard_[q_index].size() != n)
    return 1.0;
  double f;
  int i = grid_index(electron_energy_, E, f);
  const auto& v = excitation_p_hard_[q_index];
  return std::max(0.0, std::min(1.0, v(i) + f * (v(i + 1) - v(i))));
}

double Element::ionization_hard_fraction(
  int q_index, int i_shell, double E) const
{
  int n = electron_energy_.size();
  if (n < 2 ||
      ionization_p_hard_[q_index].size() != n * electroionization_.shape(0))
    return 1.0;
  double f;
  int i = grid_index(electron_energy_, E, f);
  const auto& v = ionization_p_hard_[q_index];
  return std::max(0.0,
    std::min(1.0, v(i_shell, i) + f * (v(i_shell, i + 1) - v(i_shell, i))));
}

double Element::bhabha_hard_fraction(int i_shell, double E) const
{
  int n = electron_energy_.size();
  if (n < 2 || bhabha_p_hard_.size() != n * electroionization_.shape(0))
    return 1.0;
  double f;
  int i = grid_index(electron_energy_, E, f);
  const auto& v = bhabha_p_hard_;
  return std::max(0.0,
    std::min(1.0, v(i_shell, i) + f * (v(i_shell, i + 1) - v(i_shell, i))));
}

double Element::bremsstrahlung_hard_fraction(int q_index, double E) const
{
  int n = electron_energy_.size();
  if (n < 2 || brems_p_hard_[q_index].size() != n)
    return 1.0;
  double f;
  int i = grid_index(electron_energy_, E, f);
  const auto& v = brems_p_hard_[q_index];
  return std::max(0.0, std::min(1.0, v(i) + f * (v(i + 1) - v(i))));
}

double Element::excitation(double E) const
{
  // The loss table clamps below its first abscissa, so an electron under that
  // energy would be handed a loss larger than it has
  return std::max(0.0, E - excitation_energy_loss_(E));
}

bool Element::ionization(Particle& p, int i_shell, double xi_min) const
{
  // Restricting the quantile to what a step did not group is all a hard
  // electroionization collision needs: the map from quantile to knock-on
  // energy rises, so the transfers above the cutoff are the quantiles above
  // it, and no rejection is involved.
  double density;
  double xi = xi_min + (1.0 - xi_min) * prn(p.current_seed());
  double E_knock = ionization_dist_[i_shell]->at_quantile(p.E(), xi, &density);
  // Binding energies live on shells_. Note this is NOT binding_energy_, which
  // belongs to the shorter Compton Doppler broadening shell list and would be
  // indexed out of bounds here.
  double e_b = shells_[electron_shell_map_[i_shell]].binding_energy;

  // The tabulated spectra describe a Moller collision. For a positron the
  // final state is the same -- an electron ejected with E_knock, the atom left
  // with a vacancy -- but the projectile factor is Bhabha's, so the spectrum
  // is reweighted by the ratio of the two free cross sections. Binding cancels
  // in that ratio. The cross section was raised to a majorant to pay for this
  // rejection, so declining here is a real outcome and not a lost collision.
  if (p.type().is_positron()) {
    const auto& xs {p.electron_xs(index_)};
    double majorant = moller_majorant_(xs.index_grid) +
                      xs.interp_factor * (moller_majorant_(xs.index_grid + 1) -
                                           moller_majorant_(xs.index_grid));
    FreeCollision c {p.E()};
    double x = (E_knock + e_b) / p.E();
    if (prn(p.current_seed()) * majorant >= c.ratio(x))
      return false;
  }

  // The scattered primary must be left with positive energy. The blended
  // spectrum can exceed the kinematic limit in a band roughly 0.1 eV wide just
  // above each subshell threshold, where the first tabulated table sits at
  // E = B exactly: there the limit (E-B)/2 is zero while the table still
  // carries outgoing energies. Everywhere else the blend is safe, because
  // log((E-B)/2) is concave in log E and so the geometric chord between two
  // tabulated maxima lies below the affine limit. Clamp rather than abort:
  // this is a known property of the interpolation, not a corrupt library, and
  // killing a run hours in over a 0.1 eV band would be indefensible.
  double w_max = 0.5 * (p.E() - e_b);
  if (E_knock > w_max)
    E_knock = std::max(0.0, std::nextafter(w_max, 0.0));

  double W = E_knock + e_b;
  this->emit_knock_on(p, W, e_b, this->sample_recoil(p, i_shell, W, density));
  return true;
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

double Element::sample_recoil(
  Particle& p, int i_shell, double W, double density) const
{
  constexpr double two_m = 2.0 * MASS_ELECTRON_EV;
  double E = p.E();

  // Resonance energy of the oscillator standing for this subshell. It is a
  // property of the material, not of the atom alone, because the outer shells
  // are screened by the medium they sit in.
  const auto& mat = *model::materials[p.material()];
  double w_r = mat.oscillator_energy(index_, i_shell);

  // Smallest recoil the collision can leave, reached when the projectile is
  // not deflected
  double pc = std::sqrt(E * (E + two_m));
  double pc_out = std::sqrt((E - W) * (E - W + two_m));
  double cq_min = W * (2.0 * E - W + two_m) / (pc + pc_out);
  // Q(Q + 2mc^2) = (cq)^2 inverted without subtracting nearly equal terms
  double cq_min_sq = cq_min * cq_min;
  double q_min =
    cq_min_sq / (std::sqrt(MASS_ELECTRON_EV * MASS_ELECTRON_EV + cq_min_sq) +
                  MASS_ELECTRON_EV);

  // No room below the resonance -- or no oscillator data at all -- leaves the
  // close collision as the only possibility
  if (q_min >= w_r)
    return W;

  double beta_sq =
    E * (E + two_m) / ((E + MASS_ELECTRON_EV) * (E + MASS_ELECTRON_EV));

  // Was it a close collision? The free binary cross section at this transfer
  // is known in closed form, and whatever share of the evaluated cross section
  // it accounts for is the share of collisions that struck a single electron:
  //
  //     P_close(W) = (dsigma_free/dW) / (dsigma_eval/dW)
  //
  // This is what PENELOPE's cut at the resonance energy amounts to for its own
  // delta oscillator, which places all distant strength at exactly W_i. The
  // evaluated spectra spread that strength over a range of W instead, so the
  // cut would hand close kinematics to the part of it lying above W_i, and
  // there is a good deal: for the carbon L3 shell it is a quarter of the
  // collisions where the free cross section can account for a sixteenth.
  //
  // For a positron the evaluated spectrum is reweighted by the Bhabha-to-
  // Moller ratio and the free cross section is Bhabha's, so the ratio cancels
  // out of the test and the same Moller form serves both charges -- which is
  // the same conclusion PENELOPE reaches, its distant interactions being
  // identical for the two.
  if (density > 0.0) {
    const auto& xs {p.electron_xs(index_)};
    int i_grid = xs.index_grid;
    double sigma = electroionization_(i_shell, i_grid) +
                   xs.interp_factor * (electroionization_(i_shell, i_grid + 1) -
                                        electroionization_(i_shell, i_grid));
    const auto& shell = shells_[electron_shell_map_[i_shell]];
    FreeCollision c {E};
    double sigma_free = shell.num_electrons * COLLISION_CONST / beta_sq *
                        c.moller(W / E) / (E * E);
    if (prn(p.current_seed()) * sigma * density < sigma_free)
      return W;
  } else if (W > w_r) {
    // Without the density, fall back to PENELOPE's own cut
    return W;
  }

  // Distant interaction. The transverse part is the one the density effect
  // acts on, and it hands over no momentum; the longitudinal part is
  // distributed as 1/(Q(Q + 2mc^2)) between the two bounds. The two are
  // weighted by their cross sections, whose common factor f_i / W_i cancels.
  double c_lon = std::log(w_r * (q_min + two_m) / (q_min * (w_r + two_m)));
  double c_tra =
    -std::log1p(-beta_sq) - beta_sq - mat.density_effect_correction(E);
  if (c_tra > 0.0 && prn(p.current_seed()) * (c_tra + c_lon) < c_tra)
    return q_min;

  // Invert the longitudinal distribution
  double a = std::exp(prn(p.current_seed()) * c_lon);
  return two_m * a * q_min / (q_min + two_m - a * q_min);
}

int Element::sample_ionization_shell(Particle& p, bool hard) const
{
  const auto& xs {p.electron_xs(index_)};
  int n_shell = electroionization_.shape(0);
  int i_grid = xs.index_grid;
  double f = xs.interp_factor;
  int q = p.type().is_positron() ? 1 : 0;

  // For a collision ending a condensed-history step the weights are the hard
  // parts of the shell cross sections, which differ shell by shell: the cut
  // sits at W_cc - B, so a deeply bound shell keeps more of its spectrum.
  auto weight = [&](int i) {
    double sigma =
      electroionization_(i, i_grid) +
      f * (electroionization_(i, i_grid + 1) - electroionization_(i, i_grid));
    return hard ? sigma * this->ionization_hard_fraction(q, i, p.E()) : sigma;
  };

  // Summed here rather than taken from xs.ionization, which for a positron
  // carries the majorant factor. Only the relative weights matter, and a
  // factor common to every shell cancels out of them.
  double total = 0.0;
  for (int i = 0; i < n_shell; ++i) {
    total += weight(i);
  }
  double cutoff = prn(p.current_seed()) * total;

  int i_shell;
  double prob = 0.0;
  for (i_shell = 0; i_shell < n_shell; ++i_shell) {
    // Increment probability to compare to cutoff
    prob += weight(i_shell);
    if (prob > cutoff)
      return i_shell;
  }

  // If we made it here, no shell was sampled
  p.write_restart();
  fatal_error("Did not sample any electron shell during electro-ionization.");
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
  while (true) {
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
  double ep;
  while (true) {
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
  double E_photon =
    this->sample_bremsstrahlung_energy(p.E(), p.current_seed(), k_min);
  if (E_photon <= 0.0)
    return;
  double mu = bremsstrahlung_cos_theta(Z_, p.E(), E_photon, p.current_seed());
  Direction u = rotate_angle(p.u(), mu, nullptr, p.current_seed());
  p.E() -= E_photon;
  p.create_secondary(p.wgt(), u, E_photon, ParticleType::photon());
}

} // namespace openmc
