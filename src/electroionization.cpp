#include "openmc/electroionization.h"

#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"

#include <algorithm> // for max, min, sort, unique
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

double ElectroionizationSpectrum::invert(int l, double c, double* p_local) const
{
  const auto& d {distribution_[l]};
  int n = d.e_out.size();

  // Find the last cumulative point at or below c. This runs twice per
  // ionization collision on tables of up to a few hundred points, so it is
  // worth a binary search.
  if (n < 2) {
    *p_local = (n > 0) ? d.p[0] : 0.0;
    return (n > 0) ? d.e_out[0] : 0.0;
  }
  int k = lower_bound_index(d.c.cbegin(), d.c.cend(), c);
  k = std::max(0, std::min(k, n - 2));

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
  double x =
    (m == 0.0)
      ? x_k + (c - c_k) / p_k
      : x_k +
          (std::sqrt(std::max(0.0, p_k * p_k + 2.0 * m * (c - c_k))) - p_k) / m;
  *p_local = std::max(0.0, p_k + m * (x - x_k));
  return x;
}

double ElectroionizationSpectrum::sample(
  double E, uint64_t* seed, double* density) const
{
  return this->at_quantile(E, prn(seed), density);
}

double ElectroionizationSpectrum::at_quantile(
  double E, double xi, double* density) const
{
  if (density)
    *density = 0.0;

  int n_energy = energy_.size();

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

namespace {

//! Five-point Gauss-Legendre rule on [-1, 1]
//
// The quantile map is a power blend of two cumulative inversions, each a
// square root within a tabulated interval, so it has no polynomial form -- but
// it is smooth there, and a rule exact through degree nine on each of a few
// hundred intervals leaves nothing worth chasing.
constexpr double GL_X[] = {-0.9061798459386640, -0.5384693101056831, 0.0,
  0.5384693101056831, 0.9061798459386640};
constexpr double GL_W[] = {0.2369268850561891, 0.4786286704993665,
  0.5688888888888889, 0.4786286704993665, 0.2369268850561891};

} // namespace

void ElectroionizationSpectrum::restricted_moments(double E, double e_cut,
  const std::function<double(double)>* weight, double& xi_cut, double& m0,
  double& m1, double& m2) const
{
  xi_cut = 0.0;
  m0 = 0.0;
  m1 = 0.0;
  m2 = 0.0;
  if (distribution_.empty() || e_cut <= 0.0)
    return;

  // The map from quantile to knock-on energy rises monotonically, being a
  // positive power blend of two monotone inversions, so the cutoff energy has
  // a single quantile behind it and bisection cannot land on the wrong root.
  if (this->at_quantile(E, 1.0) <= e_cut) {
    xi_cut = 1.0;
  } else if (this->at_quantile(E, 0.0) >= e_cut) {
    return;
  } else {
    double lo = 0.0;
    double hi = 1.0;
    for (int it = 0; it < 60; ++it) {
      double mid = 0.5 * (lo + hi);
      if (this->at_quantile(E, mid) <= e_cut) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    xi_cut = 0.5 * (lo + hi);
  }

  // Integrate over the quantile rather than over the energy: the map is what
  // the transport samples, and in this variable the soft region is simply
  // [0, xi_cut]. Break the range at the cumulative nodes of both tables the
  // blend uses, since the map is smooth only between them.
  vector<double> nodes;
  nodes.push_back(0.0);
  int n_energy = energy_.size();
  int i = 0;
  if (n_energy > 1) {
    double r;
    get_energy_index(energy_, E, i, r);
  }
  for (int l = i; l <= std::min(i + 1, n_energy - 1); ++l) {
    for (int k = 0; k < distribution_[l].c.size(); ++k) {
      double c = distribution_[l].c[k];
      if (c > 0.0 && c < xi_cut)
        nodes.push_back(c);
    }
  }
  nodes.push_back(xi_cut);
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

  for (int k = 0; k + 1 < nodes.size(); ++k) {
    double a = nodes[k];
    double b = nodes[k + 1];
    double half = 0.5 * (b - a);
    double mid = 0.5 * (a + b);
    for (int g = 0; g < 5; ++g) {
      double xi = mid + half * GL_X[g];
      double x = this->at_quantile(E, xi);
      double w = half * GL_W[g] * (weight ? (*weight)(x) : 1.0);
      m0 += w;
      m1 += w * x;
      m2 += w * x * x;
    }
  }
}

double ElectroionizationSpectrum::soft_quantile(double E, double e_cut) const
{
  if (distribution_.empty() || e_cut <= 0.0)
    return 0.0;
  if (this->at_quantile(E, 1.0) <= e_cut)
    return 1.0;
  if (this->at_quantile(E, 0.0) >= e_cut)
    return 0.0;

  // The map from quantile to knock-on energy rises monotonically, being a
  // positive power blend of two monotone inversions, so the cutoff has a
  // single quantile behind it and bisection cannot land on the wrong root.
  double lo = 0.0;
  double hi = 1.0;
  for (int it = 0; it < 60; ++it) {
    double mid = 0.5 * (lo + hi);
    if (this->at_quantile(E, mid) <= e_cut) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

void ElectroionizationSpectrum::integrate_quantile(double E, double xi_lo,
  double xi_hi,
  const std::function<void(double, double, double)>& accumulate) const
{
  if (distribution_.empty() || !(xi_hi > xi_lo))
    return;

  // Break the range at the cumulative nodes of both tables the blend uses,
  // since the map is smooth only between them
  vector<double> nodes;
  nodes.push_back(xi_lo);
  int n_energy = energy_.size();
  int i = 0;
  if (n_energy > 1) {
    double r;
    get_energy_index(energy_, E, i, r);
  }
  for (int l = i; l <= std::min(i + 1, n_energy - 1); ++l) {
    for (int k = 0; k < distribution_[l].c.size(); ++k) {
      double c = distribution_[l].c[k];
      if (c > xi_lo && c < xi_hi)
        nodes.push_back(c);
    }
  }
  nodes.push_back(xi_hi);
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

  for (int k = 0; k + 1 < nodes.size(); ++k) {
    double a = nodes[k];
    double b = nodes[k + 1];
    double half = 0.5 * (b - a);
    double mid = 0.5 * (a + b);
    for (int g = 0; g < 5; ++g) {
      double xi = mid + half * GL_X[g];
      double density;
      double x = this->at_quantile(E, xi, &density);
      accumulate(x, density, half * GL_W[g]);
    }
  }
}

double ElectroionizationSpectrum::integrate_quantile(double E, double xi_lo,
  double xi_hi, const std::function<double(double, double)>& f) const
{
  double total = 0.0;
  this->integrate_quantile(
    E, xi_lo, xi_hi, [&](double x, double density, double w) {
      total += w * f(x, density);
    });
  return total;
}

double ElectroionizationSpectrum::restricted_integral(
  double E, double e_cut, const std::function<double(double, double)>& f) const
{
  return this->integrate_quantile(E, 0.0, this->soft_quantile(E, e_cut), f);
}

} // namespace openmc
