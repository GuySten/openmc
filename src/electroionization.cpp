#include "openmc/electroionization.h"

#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"

#include <algorithm> // for max, min
#include <cmath>
#include <stdexcept>

namespace openmc {

ElectroionizationSpectrum::ElectroionizationSpectrum(hid_t group)
{
  read_dataset(group, "energy", energy_);
  int n_energy = energy_.size();

  vector<int> offsets;
  vector<int> interp;
  hid_t dset = open_dataset(group, "distribution");
  read_attribute(dset, "offsets", offsets);
  read_attribute(dset, "interpolation", interp);
  tensor::Tensor<double> temp;
  read_dataset(dset, temp);
  close_dataset(dset);

  for (int i = 0; i < n_energy; ++i) {
    int j = offsets[i];
    int n = (i < n_energy - 1) ? offsets[i + 1] - j : temp.shape(1) - j;

    Table t;
    t.interpolation = int2interp(interp[i]);
    t.e_out = temp.slice(0, tensor::range(j, j + n));
    t.p = temp.slice(1, tensor::range(j, j + n));
    t.c = temp.slice(2, tensor::range(j, j + n));
    distribution_.push_back(std::move(t));
  }
}

double ElectroionizationSpectrum::invert(
  int l, double c, double* p_local) const
{
  const auto& d {distribution_[l]};
  int n = d.e_out.size();

  // Find the last cumulative point at or below c
  int k = 0;
  for (int j = 0; j < n - 1; ++j) {
    if (c < d.c[j + 1])
      break;
    k = j + 1;
  }
  k = std::min(k, n - 2);

  double x_k = d.e_out[k];
  double p_k = d.p[k];
  double c_k = d.c[k];

  if (d.interpolation == Interpolation::histogram) {
    // The density is constant across the bin, so it is also the density at
    // whatever point the inversion lands on
    *p_local = p_k;
    return (p_k > 0.0) ? x_k + (c - c_k) / p_k : x_k;
  }

  if (d.interpolation != Interpolation::lin_lin) {
    throw std::runtime_error {
      "Unexpected interpolation for an electroionization spectrum."};
  }

  double x_k1 = d.e_out[k + 1];
  double p_k1 = d.p[k + 1];
  if (x_k == x_k1) {
    *p_local = p_k;
    return x_k;
  }
  double m = (p_k1 - p_k) / (x_k1 - x_k);
  double x = (m == 0.0)
               ? x_k + (c - c_k) / p_k
               : x_k +
                   (std::sqrt(std::max(0.0, p_k * p_k + 2.0 * m * (c - c_k))) -
                     p_k) /
                     m;
  *p_local = std::max(0.0, p_k + m * (x - x_k));
  return x;
}

double ElectroionizationSpectrum::sample(
  double E, uint64_t* seed, double* density) const
{
  if (density)
    *density = 0.0;

  int n_energy = energy_.size();
  double xi = prn(seed);

  // One tabulated incident energy leaves nothing to interpolate across
  if (n_energy < 2) {
    double p;
    double x = this->invert(0, xi, &p);
    if (density && p > 0.0)
      *density = p;
    return x;
  }

  int i;
  double r;
  get_energy_index(energy_, E, i, r);

  // The incident grid spans ten decades in as few as ten points -- aluminium's
  // K shell jumps from 15.8 keV to 501 keV -- so the fraction is taken
  // logarithmically. A linear one places essentially all the weight on the
  // lower table.
  double f = r;
  if (E > 0.0 && energy_[i] > 0.0 && energy_[i + 1] > energy_[i]) {
    f = std::log(E / energy_[i]) / std::log(energy_[i + 1] / energy_[i]);
    f = std::max(0.0, std::min(1.0, f));
  }

  // Both tables are inverted at the SAME quantile. Sampling each independently
  // would interpolate between two unrelated points of the two distributions
  // rather than between corresponding ones.
  double p_a, p_b;
  double x_a = this->invert(i, xi, &p_a);
  double x_b = this->invert(i + 1, xi, &p_b);

  if (!(x_a > 0.0 && x_b > 0.0))
    return x_a + f * (x_b - x_a);

  double x = x_a * std::pow(x_b / x_a, f);

  // Density of the blend. Both tables were inverted at the same quantile, so
  // the sampled value is a deterministic function of it and its density is the
  // pushforward, 1/(dx/dxi). Since xi = c_l(x_l), dx_l/dxi is 1/p_l there.
  if (density && p_a > 0.0 && p_b > 0.0) {
    double dx = x * ((1.0 - f) / (x_a * p_a) + f / (x_b * p_b));
    if (dx > 0.0)
      *density = 1.0 / dx;
  }
  return x;
}

} // namespace openmc
