#include "openmc/distribution_energy.h"

#include <algorithm> // for max, min, copy, move
#include <cmath>     // for sqrt, abs, log, pow
#include <cstddef>   // for size_t
#include <iterator>  // for back_inserter

#include "openmc/tensor.h"

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/particle.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"

namespace openmc {

//==============================================================================
// DiscretePhoton implementation
//==============================================================================

DiscretePhoton::DiscretePhoton(hid_t group)
{
  read_attribute(group, "primary_flag", primary_flag_);
  read_attribute(group, "energy", energy_);
  read_attribute(group, "atomic_weight_ratio", A_);
}

double DiscretePhoton::sample(double E, uint64_t* seed) const
{
  if (primary_flag_ == 2) {
    return energy_ + A_ / (A_ + 1) * E;
  } else {
    return energy_;
  }
}

//==============================================================================
// LevelInelastic implementation
//==============================================================================

LevelInelastic::LevelInelastic(hid_t group)
{
  // for backwards compatibility:
  if (attribute_exists(group, "mass_ratio")) {
    read_attribute(group, "threshold", b_);
    read_attribute(group, "mass_ratio", a_);
    c_ = 0.0;
  } else {
    double A, Q;
    std::string temp;
    read_attribute(group, "mass", A);
    read_attribute(group, "q_value", Q);
    read_attribute(group, "particle", temp);
    auto type = ParticleType(temp);
    if (type.is_neutron()) {
      a_ = (A / (A + 1.0)) * (A / (A + 1.0));
      b_ = (A + 1.0) / A * std::abs(Q);
      c_ = 0.0;
    } else if (type.is_photon()) {
      a_ = (A - 1.0) / A;
      b_ = std::abs(Q);
      c_ = 1.0 / (2.0 * MASS_NEUTRON_EV * A);
    } else {
      fatal_error("Unrecognized particle: " + type.str());
    }
  }
}

double LevelInelastic::sample(double E, uint64_t* seed) const
{
  return a_ * (E - b_ - c_ * (E * E));
}

//==============================================================================
// ContinuousTabular implementation
//==============================================================================

ContinuousTabular::ContinuousTabular(hid_t group)
{
  // Open incoming energy dataset
  hid_t dset = open_dataset(group, "energy");

  // Get interpolation parameters
  tensor::Tensor<int> temp;
  read_attribute(dset, "interpolation", temp);

  tensor::View<int> temp_b = temp.slice(0); // breakpoints
  tensor::View<int> temp_i = temp.slice(1); // interpolation parameters

  std::copy(temp_b.begin(), temp_b.end(), std::back_inserter(breakpoints_));
  for (const auto i : temp_i)
    interpolation_.push_back(int2interp(i));
  n_region_ = breakpoints_.size();

  // Get incoming energies
  read_dataset(dset, energy_);
  std::size_t n_energy = energy_.size();
  close_dataset(dset);

  // Get outgoing energy distribution data
  dset = open_dataset(group, "distribution");
  vector<int> offsets;
  vector<int> interp;
  vector<int> n_discrete;
  read_attribute(dset, "offsets", offsets);
  read_attribute(dset, "interpolation", interp);
  read_attribute(dset, "n_discrete_lines", n_discrete);

  tensor::Tensor<double> eout;
  read_dataset(dset, eout);
  close_dataset(dset);

  for (int i = 0; i < n_energy; ++i) {
    // Determine number of outgoing energies
    int j = offsets[i];
    int n;
    if (i < n_energy - 1) {
      n = offsets[i + 1] - j;
    } else {
      n = eout.shape(1) - j;
    }

    // Assign interpolation scheme and number of discrete lines
    CTTable d;
    d.interpolation = int2interp(interp[i]);
    d.n_discrete = n_discrete[i];

    // Copy data
    d.e_out = eout.slice(0, tensor::range(j, j + n));
    d.p = eout.slice(1, tensor::range(j, j + n));

    // To get answers that match ACE data, for now we still use the tabulated
    // CDF values that were passed through to the HDF5 library. At a later
    // time, we can remove the CDF values from the HDF5 library and
    // reconstruct them using the PDF
    if (true) {
      d.c = eout.slice(2, tensor::range(j, j + n));
    } else {
      // Calculate cumulative distribution function -- discrete portion
      for (int k = 0; k < d.n_discrete; ++k) {
        if (k == 0) {
          d.c[k] = d.p[k];
        } else {
          d.c[k] = d.c[k - 1] + d.p[k];
        }
      }

      // Continuous portion
      for (int k = d.n_discrete; k < n; ++k) {
        if (k == d.n_discrete) {
          d.c[k] = d.c[k - 1] + d.p[k];
        } else {
          if (d.interpolation == Interpolation::histogram) {
            d.c[k] = d.c[k - 1] + d.p[k - 1] * (d.e_out[k] - d.e_out[k - 1]);
          } else if (d.interpolation == Interpolation::lin_lin) {
            d.c[k] = d.c[k - 1] + 0.5 * (d.p[k - 1] + d.p[k]) *
                                    (d.e_out[k] - d.e_out[k - 1]);
          }
        }
      }

      // Normalize density and distribution functions
      d.p /= d.c[n - 1];
      d.c /= d.c[n - 1];
    }

    distribution_.push_back(std::move(d));
  } // incoming energies
}

double ContinuousTabular::sample_table(int l, double r1, bool& discrete) const
{
  const auto& d {distribution_[l]};
  int n_energy_out = d.e_out.size();
  int n_discrete = d.n_discrete;
  double c_k = d.c[0];
  int k = 0;
  int end = n_energy_out - 2;

  // Discrete portion
  for (int j = 0; j < n_discrete; ++j) {
    k = j;
    c_k = d.c[k];
    if (r1 < c_k) {
      end = j;
      break;
    }
  }

  // Continuous portion
  double c_k1;
  for (int j = n_discrete; j < end; ++j) {
    k = j;
    c_k1 = d.c[k + 1];
    if (r1 < c_k1)
      break;
    k = j + 1;
    c_k = c_k1;
  }

  discrete = (k < n_discrete);

  double E_l_k = d.e_out[k];
  double p_l_k = d.p[k];
  double E_out = E_l_k;

  if (d.interpolation == Interpolation::histogram) {
    if (p_l_k > 0.0 && k >= n_discrete) {
      E_out = E_l_k + (r1 - c_k) / p_l_k;
    }

  } else if (d.interpolation == Interpolation::lin_lin) {
    double E_l_k1 = d.e_out[k + 1];
    double p_l_k1 = d.p[k + 1];

    if (E_l_k != E_l_k1) {
      double frac = (p_l_k1 - p_l_k) / (E_l_k1 - E_l_k);
      if (frac == 0.0) {
        E_out = E_l_k + (r1 - c_k) / p_l_k;
      } else {
        E_out =
          E_l_k +
          (std::sqrt(std::max(0.0, p_l_k * p_l_k + 2.0 * frac * (r1 - c_k))) -
            p_l_k) /
            frac;
      }
    }
  } else {
    throw std::runtime_error {"Unexpected interpolation for continuous energy "
                              "distribution."};
  }

  return E_out;
}

double ContinuousTabular::sample(double E, uint64_t* seed) const
{
  // Read number of interpolation regions and incoming energies
  bool histogram_interp;
  bool loglog_interp;
  if (n_region_ == 1) {
    histogram_interp = (interpolation_[0] == Interpolation::histogram);
    loglog_interp = (interpolation_[0] == Interpolation::log_log);
  } else {
    histogram_interp = false;
    loglog_interp = false;
  }

  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  auto n_energy_in = energy_.size();
  int i;
  double r;
  if (E < energy_[0]) {
    i = 0;
    r = 0.0;
  } else if (E > energy_[n_energy_in - 1]) {
    i = n_energy_in - 2;
    r = 1.0;
  } else {
    i = lower_bound_index(energy_.begin(), energy_.end(), E);
    // With log-log interpolation the incident energy fraction is taken
    // logarithmically. This matters when the incident energy grid is sparse:
    // the EEDL-derived bremsstrahlung tables span ten decades in as few as ten
    // points, where a linear fraction places essentially all the weight on the
    // lower table.
    if (loglog_interp && E > 0.0 && energy_[i] > 0.0 &&
        energy_[i + 1] > energy_[i]) {
      r = std::log(E / energy_[i]) / std::log(energy_[i + 1] / energy_[i]);
    } else {
      r = (E - energy_[i]) / (energy_[i + 1] - energy_[i]);
    }
  }

  double r1 = prn(seed);

  if (histogram_interp) {
    bool discrete;
    return this->sample_table(i, r1, discrete);
  }

  // Bounds of the scaled outgoing energy range for each bracketing table
  auto bounds = [&](int l, double& E_1, double& E_K) {
    int n = distribution_[l].e_out.size();
    E_1 = distribution_[l].e_out[distribution_[l].n_discrete];
    E_K = distribution_[l].e_out[n - 1];
  };

  double E_i_1, E_i_K, E_i1_1, E_i1_K;
  bounds(i, E_i_1, E_i_K);
  bounds(i + 1, E_i1_1, E_i1_K);

  // The scaled interpolation bounds must be blended with the same scheme used
  // for r, or the endpoint is wrong. For a quantity proportional to the
  // incident energy, geometric blending with a logarithmic r reproduces the
  // incident energy exactly, just as linear blending does with a linear r.
  double E_1;
  double E_K;
  if (loglog_interp && E_i_1 > 0.0 && E_i1_1 > 0.0 && E_i_K > 0.0 &&
      E_i1_K > 0.0) {
    E_1 = E_i_1 * std::pow(E_i1_1 / E_i_1, r);
    E_K = E_i_K * std::pow(E_i1_K / E_i_K, r);
  } else {
    E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
    E_K = E_i_K + r * (E_i1_K - E_i_K);
  }

  if (loglog_interp) {
    // Interpolate the distributions rather than choosing between them. Both
    // tables are inverted at the same cumulative probability and their results
    // mapped onto a common unit scale, which interpolates the inverse CDF and
    // therefore the distribution itself. Selecting one table stochastically,
    // as the linear path below does, instead yields a mixture of the two
    // shapes -- adequate on a dense grid, but not across the decade-wide gaps
    // in the EEDL tables.
    bool discrete_i, discrete_i1;
    double E_out_i = this->sample_table(i, r1, discrete_i);
    double E_out_i1 = this->sample_table(i + 1, r1, discrete_i1);

    // Discrete lines are returned unscaled and cannot be blended; fall back to
    // stochastic selection for them.
    if (discrete_i || discrete_i1) {
      return (r > prn(seed)) ? E_out_i1 : E_out_i;
    }

    double span_i = E_i_K - E_i_1;
    double span_i1 = E_i1_K - E_i1_1;
    if (span_i <= 0.0 || span_i1 <= 0.0)
      return E_out_i;

    double x_i = (E_out_i - E_i_1) / span_i;
    double x_i1 = (E_out_i1 - E_i1_1) / span_i1;
    double x = x_i + r * (x_i1 - x_i);

    return E_1 + x * (E_K - E_1);
  }

  // Sample between the ith and [i+1]th bin
  int l = r > prn(seed) ? i + 1 : i;

  bool discrete;
  double E_out = this->sample_table(l, r1, discrete);

  // Now interpolate between incident energy bins i and i + 1
  if (!discrete && distribution_[l].e_out.size() > 1) {
    if (l == i) {
      return E_1 + (E_out - E_i_1) * (E_K - E_1) / (E_i_K - E_i_1);
    } else {
      return E_1 + (E_out - E_i1_1) * (E_K - E_1) / (E_i1_K - E_i1_1);
    }
  } else {
    return E_out;
  }
}

//==============================================================================
// MaxwellEnergy implementation
//==============================================================================

MaxwellEnergy::MaxwellEnergy(hid_t group)
{
  read_attribute(group, "u", u_);
  hid_t dset = open_dataset(group, "theta");
  theta_ = Tabulated1D {dset};
  close_dataset(dset);
}

double MaxwellEnergy::sample(double E, uint64_t* seed) const
{
  // Get temperature corresponding to incoming energy
  double theta = theta_(E);

  while (true) {
    // Sample maxwell fission spectrum
    double E_out = maxwell_spectrum(theta, seed);

    // Accept energy based on restriction energy
    if (E_out <= E - u_)
      return E_out;
  }
}

//==============================================================================
// Evaporation implementation
//==============================================================================

Evaporation::Evaporation(hid_t group)
{
  read_attribute(group, "u", u_);
  hid_t dset = open_dataset(group, "theta");
  theta_ = Tabulated1D {dset};
  close_dataset(dset);
}

double Evaporation::sample(double E, uint64_t* seed) const
{
  // Get temperature corresponding to incoming energy
  double theta = theta_(E);

  double y = (E - u_) / theta;
  double v = 1.0 - std::exp(-y);

  // Sample outgoing energy based on evaporation spectrum probability
  // density function
  double x;
  while (true) {
    x = -std::log((1.0 - v * prn(seed)) * (1.0 - v * prn(seed)));
    if (x <= y)
      break;
  }

  return x * theta;
}

//==============================================================================
// WattEnergy implementation
//==============================================================================

WattEnergy::WattEnergy(hid_t group)
{
  // Read restriction energy
  read_attribute(group, "u", u_);

  // Read tabulated functions
  hid_t dset = open_dataset(group, "a");
  a_ = Tabulated1D {dset};
  close_dataset(dset);
  dset = open_dataset(group, "b");
  b_ = Tabulated1D {dset};
  close_dataset(dset);
}

double WattEnergy::sample(double E, uint64_t* seed) const
{
  // Determine Watt parameters at incident energy
  double a = a_(E);
  double b = b_(E);

  while (true) {
    // Sample energy-dependent Watt fission spectrum
    double E_out = watt_spectrum(a, b, seed);

    // Accept energy based on restriction energy
    if (E_out <= E - u_)
      return E_out;
  }
}

//==============================================================================
// max_energy implementations
//==============================================================================

double DiscretePhoton::max_energy(double E) const
{
  return primary_flag_ == 2 ? energy_ + A_ / (A_ + 1) * E : energy_;
}

double LevelInelastic::max_energy(double E) const
{
  // Deterministic: sample() has no random component
  return a_ * (E - b_ - c_ * (E * E));
}

double ContinuousTabular::max_energy(double E) const
{
  bool histogram_interp =
    (n_region_ == 1) && (interpolation_[0] == Interpolation::histogram);

  // A single tabulated incident energy leaves no bin to interpolate across,
  // and distribution_[i + 1] below would be out of bounds.
  auto n_energy_in = energy_.size();
  if (n_energy_in < 2) {
    const auto& d = distribution_[0];
    return d.e_out[d.e_out.size() - 1];
  }

  // Find energy bin and interpolation factor exactly as sample() does
  int i;
  double r;
  if (E < energy_[0]) {
    i = 0;
    r = 0.0;
  } else if (E > energy_[n_energy_in - 1]) {
    i = n_energy_in - 2;
    r = 1.0;
  } else {
    i = lower_bound_index(energy_.begin(), energy_.end(), E);
    r = (E - energy_[i]) / (energy_[i + 1] - energy_[i]);
  }

  if (histogram_interp) {
    const auto& d = distribution_[i];
    return d.e_out[d.e_out.size() - 1];
  }

  // Continuous portion is scaled onto [E_1, E_K]; E_K is the attainable
  // maximum. Discrete lines are returned unscaled, so they are bounded
  // separately.
  const auto& d_i = distribution_[i];
  const auto& d_i1 = distribution_[i + 1];
  double E_i_K = d_i.e_out[d_i.e_out.size() - 1];
  double E_i1_K = d_i1.e_out[d_i1.e_out.size() - 1];
  double result = E_i_K + r * (E_i1_K - E_i_K);

  for (const auto* d : {&d_i, &d_i1}) {
    for (int j = 0; j < d->n_discrete; ++j) {
      result = std::max(result, d->e_out[j]);
    }
  }
  return result;
}

double Evaporation::max_energy(double E) const
{
  // sample() rejects until x <= y = (E - u_)/theta, so E_out <= E - u_
  return E - u_;
}

double MaxwellEnergy::max_energy(double E) const
{
  return E - u_;
}

double WattEnergy::max_energy(double E) const
{
  return E - u_;
}

} // namespace openmc
