#include "openmc/distribution_energy.h"

#include <algorithm> // for max, min, copy, move
#include <cmath>     // for sqrt, abs
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

void ContinuousTabular::select_table(
  double E, uint64_t* seed, int& i, double& r, int& l, bool& hist) const
{
  // Read number of interpolation regions and incoming energies
  if (n_region_ == 1) {
    hist = (interpolation_[0] == Interpolation::histogram);
  } else {
    hist = false;
  }

  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  auto n_energy_in = energy_.size();
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

  // Sample between the ith and [i+1]th bin
  if (hist) {
    l = i;
  } else {
    l = r > prn(seed) ? i + 1 : i;
  }
}

double ContinuousTabular::invert_cdf(
  int i, double r, int l, double r1, bool hist) const
{
  // Determine outgoing energy bin
  int n_energy_out = distribution_[l].e_out.size();
  int n_discrete = distribution_[l].n_discrete;
  double c_k = distribution_[l].c[0];
  int k = 0;
  int end = n_energy_out - 2;

  // Discrete portion
  for (int j = 0; j < n_discrete; ++j) {
    k = j;
    c_k = distribution_[l].c[k];
    if (r1 < c_k) {
      end = j;
      break;
    }
  }

  // Continuous portion
  double c_k1;
  for (int j = n_discrete; j < end; ++j) {
    k = j;
    c_k1 = distribution_[l].c[k + 1];
    if (r1 < c_k1)
      break;
    k = j + 1;
    c_k = c_k1;
  }

  double E_l_k = distribution_[l].e_out[k];

  if (k < n_discrete) {
    // Discrete case
    return E_l_k;
  } else {
    // Continuous case
    double p_l_k = distribution_[l].p[k];
    double E_out;
    if (distribution_[l].interpolation == Interpolation::histogram) {
      // Histogram interpolation
      if (p_l_k > 0.0) {
        E_out = E_l_k + (r1 - c_k) / p_l_k;
      } else {
        E_out = E_l_k;
      }
    } else if (distribution_[l].interpolation == Interpolation::lin_lin) {
      // Linear-linear interpolation
      double E_l_k1 = distribution_[l].e_out[k + 1];
      double p_l_k1 = distribution_[l].p[k + 1];

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
      } else {
        E_out = E_l_k;
      }
    } else {
      throw std::runtime_error {
        "Unexpected interpolation for continuous energy "
        "distribution."};
    }

    // Now interpolate between incident energy bins i and i + 1
    if (!hist && n_energy_out > 1) {
      // Interpolation for energy E1 and EK
      n_energy_out = distribution_[i].e_out.size();
      n_discrete = distribution_[i].n_discrete;
      const double E_i_1 = distribution_[i].e_out[n_discrete];
      const double E_i_K = distribution_[i].e_out[n_energy_out - 1];

      n_energy_out = distribution_[i + 1].e_out.size();
      n_discrete = distribution_[i + 1].n_discrete;
      const double E_i1_1 = distribution_[i + 1].e_out[n_discrete];
      const double E_i1_K = distribution_[i + 1].e_out[n_energy_out - 1];

      const double E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
      const double E_K = E_i_K + r * (E_i1_K - E_i_K);

      if (l == i) {
        return E_1 + (E_out - E_i_1) * (E_K - E_1) / (E_i_K - E_i_1);
      } else {
        return E_1 + (E_out - E_i1_1) * (E_K - E_1) / (E_i1_K - E_i1_1);
      }
    } else {
      return E_out;
    }
  }
}

double ContinuousTabular::sample(double E, uint64_t* seed) const
{
  int i;
  int l;
  double r;
  bool hist;
  select_table(E, seed, i, r, l, hist);
  return invert_cdf(i, r, l, prn(seed), hist);
}

double ContinuousTabular::sample_above(
  double E, double E_min, uint64_t* seed, double& E_out) const
{
  int i;
  int l;
  double r;
  bool hist;
  select_table(E, seed, i, r, l, hist);

  const auto& dist = distribution_[l];
  const int n_out = dist.e_out.size();
  const int n_disc = dist.n_discrete;

  // Express the threshold in this table's own outgoing-energy coordinates.
  // invert_cdf() maps the continuous part of table l onto a range
  // interpolated between tables i and i + 1, so the threshold has to be
  // mapped back through that same straight line before it can be compared
  // against e_out.
  double E_min_l = E_min;
  if (!hist && n_out > 1) {
    const int nd_i = distribution_[i].n_discrete;
    const double E_i_1 = distribution_[i].e_out[nd_i];
    const double E_i_K =
      distribution_[i].e_out[distribution_[i].e_out.size() - 1];

    const int nd_i1 = distribution_[i + 1].n_discrete;
    const double E_i1_1 = distribution_[i + 1].e_out[nd_i1];
    const double E_i1_K =
      distribution_[i + 1].e_out[distribution_[i + 1].e_out.size() - 1];

    const double E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
    const double E_K = E_i_K + r * (E_i1_K - E_i_K);

    if (E_K <= E_1) {
      // Degenerate range; nothing to map onto, so do not restrict.
      E_out = invert_cdf(i, r, l, prn(seed), hist);
      return 1.0;
    }

    const double E_l_1 = (l == i) ? E_i_1 : E_i1_1;
    const double E_l_K = (l == i) ? E_i_K : E_i1_K;
    E_min_l = E_l_1 + (E_min - E_1) * (E_l_K - E_l_1) / (E_K - E_1);
  }

  // The last tabulated point at or below the threshold. Cutting there rather
  // than at the threshold itself keeps this to a grid lookup with no CDF
  // interpolation; the sliver it leaves in, between that point and the
  // threshold, is discarded by the caller and so costs nothing but a few
  // wasted draws.
  int k_lo = n_disc;
  while (k_lo + 1 < n_out && dist.e_out[k_lo + 1] <= E_min_l)
    ++k_lo;

  // Every discrete line is kept whatever its energy. They come first in the
  // array whatever their energies are, so they are not a contiguous range of
  // the CDF alongside the continuous tail, and dropping the ones below the
  // threshold by a single cut would drop the ones above it too. Keeping them
  // all is what makes this a restriction that can only ever include extra,
  // never exclude what belongs.
  const double c_disc = (n_disc > 0) ? dist.c[n_disc - 1] : 0.0;
  const double c_lo = std::max(dist.c[k_lo], c_disc);
  const double mass = c_disc + (1.0 - c_lo);

  if (mass <= 0.0) {
    // Nothing at all at or above the threshold.
    E_out = 0.0;
    return 0.0;
  }

  // One random number, exactly as sample() draws, mapped onto the retained
  // part of the CDF: the discrete lines below c_disc, then the tail from
  // c_lo up.
  const double u = mass * prn(seed);
  const double r1 = (u < c_disc) ? u : (u - c_disc + c_lo);

  E_out = invert_cdf(i, r, l, r1, hist);
  return mass;
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
