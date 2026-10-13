#include "openmc/distribution_energy.h"

#include <algorithm> // for max, min, copy, move
#include <cmath>     // for erf, exp, sqrt
#include <cstddef>   // for size_t
#include <iterator>  // for back_inserter

#include "openmc/tensor.h"

#include "openmc/constants.h"
#include "openmc/endf.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
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
  read_attribute(group, "threshold", threshold_);
  read_attribute(group, "mass_ratio", mass_ratio_);
}

double LevelInelastic::sample(double E, uint64_t* seed) const
{
  return mass_ratio_ * (E - threshold_);
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

double ContinuousTabular::sample(double E, uint64_t* seed) const
{
  // Read number of interpolation regions and incoming energies
  bool histogram_interp;
  if (n_region_ == 1) {
    histogram_interp = (interpolation_[0] == Interpolation::histogram);
  } else {
    histogram_interp = false;
  }

  // Find energy bin and calculate interpolation factor -- if the energy is
  // outside the range of the tabulated energies, choose the first or last bins
  int i;
  double r;
  get_energy_index(energy_, E, i, r);

  // Sample between the ith and [i+1]th bin
  int l;
  if (histogram_interp) {
    l = i;
  } else {
    l = r > prn(seed) ? i + 1 : i;
  }

  // Determine outgoing energy bin
  int n_energy_out = distribution_[l].e_out.size();
  int n_discrete = distribution_[l].n_discrete;
  double r1 = prn(seed);
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
    if (!histogram_interp && n_energy_out > 1) {
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

double ContinuousTabular::table_pdf(int l, double E_out) const
{
  const auto& d = distribution_[l];

  // Discrete lines have no density
  if (d.n_discrete > 0)
    return -1.0;

  int n = d.e_out.size();
  if (n < 2 || E_out < d.e_out[0] || E_out > d.e_out[n - 1])
    return 0.0;

  // Find the outgoing energy bin containing E_out
  int k = lower_bound_index(d.e_out.begin(), d.e_out.end(), E_out);
  k = std::max(0, std::min(k, n - 2));

  if (d.interpolation == Interpolation::histogram) {
    return d.p[k];
  } else if (d.interpolation == Interpolation::lin_lin) {
    double dE = d.e_out[k + 1] - d.e_out[k];
    if (dE == 0.0)
      return d.p[k];
    return d.p[k] + (d.p[k + 1] - d.p[k]) * (E_out - d.e_out[k]) / dE;
  }
  return -1.0;
}

double ContinuousTabular::pdf(double E, double E_out) const
{
  // This mirrors sample(): table l is chosen with probability 1 - r (l = i)
  // or r (l = i + 1) and its outgoing energy is then mapped linearly onto the
  // interpolated range [E_1, E_K], which scales the density by the inverse of
  // the stretch factor
  bool histogram_interp;
  if (n_region_ == 1) {
    histogram_interp = (interpolation_[0] == Interpolation::histogram);
  } else {
    histogram_interp = false;
  }

  int i;
  double r;
  get_energy_index(energy_, E, i, r);

  if (histogram_interp) {
    return table_pdf(i, E_out);
  }

  // Bounds of the continuous part of the two bracketing tables
  auto bounds = [this](int l, double& E_lo, double& E_hi) {
    const auto& d = distribution_[l];
    E_lo = d.e_out[d.n_discrete];
    E_hi = d.e_out[d.e_out.size() - 1];
  };

  double E_i_1, E_i_K;
  bounds(i, E_i_1, E_i_K);
  double E_i1_1 = E_i_1;
  double E_i1_K = E_i_K;
  if (i + 1 < static_cast<int>(distribution_.size())) {
    bounds(i + 1, E_i1_1, E_i1_K);
  }
  const double E_1 = E_i_1 + r * (E_i1_1 - E_i_1);
  const double E_K = E_i_K + r * (E_i1_K - E_i_K);

  double density = 0.0;
  for (int m = 0; m < 2; ++m) {
    int l = i + m;
    double prob = (m == 0) ? 1.0 - r : r;
    if (prob <= 0.0)
      continue;

    if (distribution_[l].e_out.size() <= 1) {
      double f = table_pdf(l, E_out);
      if (f < 0.0)
        return -1.0;
      density += prob * f;
      continue;
    }

    double E_l_1 = (m == 0) ? E_i_1 : E_i1_1;
    double E_l_K = (m == 0) ? E_i_K : E_i1_K;
    double stretch = (E_K - E_1) / (E_l_K - E_l_1);
    if (stretch <= 0.0)
      return -1.0;
    double E_table = E_l_1 + (E_out - E_1) / stretch;
    double f = table_pdf(l, E_table);
    if (f < 0.0)
      return -1.0;
    density += prob * f / stretch;
  }
  return density;
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

double MaxwellEnergy::pdf(double E, double E_out) const
{
  double theta = theta_(E);
  double E_max = E - u_;
  if (E_out < 0.0 || E_out > E_max || theta <= 0.0)
    return 0.0;

  // Maxwellian density truncated at the restriction energy
  double x = E_max / theta;
  double norm = std::erf(std::sqrt(x)) - 2.0 * std::sqrt(x / PI) * std::exp(-x);
  return 2.0 / std::sqrt(PI) * std::sqrt(E_out) / std::pow(theta, 1.5) *
         std::exp(-E_out / theta) / norm;
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

double Evaporation::pdf(double E, double E_out) const
{
  double theta = theta_(E);
  double E_max = E - u_;
  if (E_out < 0.0 || E_out > E_max || theta <= 0.0)
    return 0.0;

  // Evaporation density truncated at the restriction energy
  double y = E_max / theta;
  double norm = theta * theta * (1.0 - std::exp(-y) * (1.0 + y));
  return E_out * std::exp(-E_out / theta) / norm;
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

double WattEnergy::pdf(double E, double E_out) const
{
  double a = a_(E);
  double b = b_(E);
  double E_max = E - u_;
  if (E_out < 0.0 || E_out > E_max || a <= 0.0 || b <= 0.0)
    return 0.0;

  // Integral of exp(-E/a) sinh(sqrt(b E)) from 0 to E_max, obtained by
  // substituting s = sqrt(E) and completing the square in each exponential
  double S = std::sqrt(E_max);
  double c = 0.5 * a * std::sqrt(b);
  double sa = std::sqrt(a);
  double norm =
    std::exp(0.25 * a * b) *
    (0.5 * a *
        (std::exp(-(S + c) * (S + c) / a) - std::exp(-(S - c) * (S - c) / a)) +
      0.5 * c * std::sqrt(PI * a) *
        (std::erf((S - c) / sa) + std::erf((S + c) / sa)));
  return std::exp(-E_out / a) * std::sinh(std::sqrt(b * E_out)) / norm;
}

} // namespace openmc
