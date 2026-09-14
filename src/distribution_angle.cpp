#include "openmc/distribution_angle.h"

#include <cmath> // for abs, copysign

#include "openmc/tensor.h"

#include "openmc/endf.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/vector.h" // for vector

namespace openmc {

//==============================================================================
// AngleDistribution implementation
//==============================================================================

AngleDistribution::AngleDistribution(hid_t group)
{
  // Get incoming energies
  read_dataset(group, "energy", energy_);
  int n_energy = energy_.size();

  // Get outgoing energy distribution data
  vector<int> offsets;
  vector<int> interp;
  hid_t dset = open_dataset(group, "mu");
  read_attribute(dset, "offsets", offsets);
  read_attribute(dset, "interpolation", interp);
  tensor::Tensor<double> temp;
  read_dataset(dset, temp);
  close_dataset(dset);

  for (int i = 0; i < n_energy; ++i) {
    // Determine number of outgoing energies
    int j = offsets[i];
    int n;
    if (i < n_energy - 1) {
      n = offsets[i + 1] - j;
    } else {
      n = temp.shape(1) - j;
    }

    // Create and initialize tabular distribution
    tensor::View<double> xs = temp.slice(0, tensor::range(j, j + n));
    tensor::View<double> ps = temp.slice(1, tensor::range(j, j + n));
    tensor::View<double> cs = temp.slice(2, tensor::range(j, j + n));
    vector<double> x {xs.begin(), xs.end()};
    vector<double> p {ps.begin(), ps.end()};
    vector<double> c {cs.begin(), cs.end()};

    // To get answers that match ACE data, for now we still use the tabulated
    // CDF values that were passed through to the HDF5 library. At a later
    // time, we can remove the CDF values from the HDF5 library and
    // reconstruct them using the PDF
    Tabular* mudist =
      new Tabular {x.data(), p.data(), n, int2interp(interp[i]), c.data()};

    distribution_.emplace_back(mudist);
  }
}

double AngleDistribution::sample(double E, uint64_t* seed) const
{
  // Find energy bin and calculate interpolation factor
  int i;
  double r;
  get_energy_index(energy_, E, i, r);

  // Sample between the ith and (i+1)th bin
  if (r > prn(seed))
    ++i;

  // Sample i-th distribution
  double mu = distribution_[i]->sample(seed).first;

  // Make sure mu is in range [-1,1] and return
  if (std::abs(mu) > 1.0)
    mu = std::copysign(1.0, mu);
  return mu;
}

double AngleDistribution::sample_log_interp(double E, uint64_t* seed) const
{
  auto n = energy_.size();
  if (n == 1)
    return distribution_[0]->sample_at(prn(seed));

  // Bracket the energy without the interpolation factor sample() would use
  int i;
  if (E <= energy_[0]) {
    i = 0;
  } else if (E >= energy_[n - 1]) {
    i = n - 2;
  } else {
    i = lower_bound_index(energy_.begin(), energy_.end(), E);
  }

  // Fraction in log-energy, which is the variable these tables are actually
  // spaced on -- they step geometrically, so a linear fraction is meaningless
  // across a decade-wide interval.
  double f = std::log(E / energy_[i]) / std::log(energy_[i + 1] / energy_[i]);
  f = std::max(0.0, std::min(1.0, f));

  // Sample BOTH bracketing tables at the same quantile and interpolate the
  // result, rather than picking one table at random. Choosing a table with
  // probability f reproduces the linear average of the two distributions; for
  // elastic scattering the deflection scales roughly as 1/E between tables, so
  // the linear average is dominated by the low-energy table right across the
  // interval and over-scatters by a large factor.
  double xi = prn(seed);
  double mu_low = distribution_[i]->sample_at(xi);
  double mu_high = distribution_[i + 1]->sample_at(xi);

  // Interpolate the deflection (1-mu) geometrically. That is the quantity with
  // the power-law energy dependence, so a geometric interpolation of it
  // reproduces the trend exactly when the two tables share a shape; a linear
  // interpolation of mu would not.
  double d_low = 1.0 - mu_low;
  double d_high = 1.0 - mu_high;
  double mu;
  if (d_low > 0.0 && d_high > 0.0) {
    mu = 1.0 - std::exp((1.0 - f) * std::log(d_low) + f * std::log(d_high));
  } else {
    // One of the samples is exactly forward; nothing to interpolate
    // geometrically, so fall back to a linear blend.
    mu = (1.0 - f) * mu_low + f * mu_high;
  }

  if (std::abs(mu) > 1.0)
    mu = std::copysign(1.0, mu);
  return mu;
}

double AngleDistribution::evaluate(double E, double mu) const
{
  // Find energy bin and calculate interpolation factor
  int i;
  double r;
  get_energy_index(energy_, E, i, r);

  double pdf = 0.0;
  if (r > 0.0)
    pdf += r * distribution_[i + 1]->evaluate(mu);
  if (r < 1.0)
    pdf += (1.0 - r) * distribution_[i]->evaluate(mu);
  return pdf;
}

} // namespace openmc
