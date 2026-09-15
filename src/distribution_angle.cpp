#include "openmc/distribution_angle.h"

#include <algorithm> // for sort, unique
#include <cmath>     // for abs, copysign

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

AngleDistribution::AngleDistribution(hid_t group, Interpolation energy_interp)
  : energy_interp_ {energy_interp}
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

  double mu;
  if (energy_interp_ == Interpolation::log_log && energy_.size() > 1) {
    // Fraction in log-energy, which is the variable these tables are spaced
    // on. A linear fraction is close to meaningless across an interval that
    // spans a decade or more.
    double f = std::log(E / energy_[i]) / std::log(energy_[i + 1] / energy_[i]);
    f = std::max(0.0, std::min(1.0, f));

    // Sample both bracketing tables at the SAME quantile. Sampling each
    // independently would interpolate between two unrelated points of the two
    // distributions rather than between corresponding ones.
    //
    // lin_lin, the default, is approximated the way OpenMC always has: choose
    // one table at random with a probability linear in energy. That reproduces
    // the linear average of the two distributions, which is adequate where
    // they are closely spaced.
    double xi = prn(seed);
    double mu_low = distribution_[i]->sample_at(xi);
    double mu_high = distribution_[i + 1]->sample_at(xi);

    // Interpolate the deflection 1-mu geometrically: that is the quantity
    // following a power of the energy, so this reproduces the trend exactly
    // where the two tables share a shape. Interpolating mu linearly would not,
    // and choosing one table at random reproduces the linear average of the
    // two distributions, which a power law dominates from its low end.
    double d_low = 1.0 - mu_low;
    double d_high = 1.0 - mu_high;
    if (d_low > 0.0 && d_high > 0.0) {
      mu = 1.0 - std::exp((1.0 - f) * std::log(d_low) + f * std::log(d_high));
    } else {
      // A sample landed exactly forward, leaving nothing to interpolate
      // geometrically.
      mu = (1.0 - f) * mu_low + f * mu_high;
    }
  } else {
    // Sample between the ith and (i+1)th bin
    if (r > prn(seed))
      ++i;

    // Sample i-th distribution
    mu = distribution_[i]->sample(seed).first;
  }

  // Make sure mu is in range [-1,1] and return
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
  return r * distribution_[i + 1]->evaluate(mu) +
         (1.0 - r) * distribution_[i]->evaluate(mu);
}

} // namespace openmc
