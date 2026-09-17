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

namespace {

//! Integrals of p, p(1-mu) and (3/2) p (1-mu^2) over part of one tabulated
//! segment, exact for the interpolation law in force.
struct SegmentMoments {
  double norm;
  double m1;
  double m2;
};

//! \param[in] x0 Left end of the segment
//! \param[in] a Density at x0
//! \param[in] m Slope of the density in u = mu - x0, zero for a histogram
//! \param[in] u Upper limit of the integration, measured from x0
SegmentMoments segment_moments(double x0, double a, double m, double u)
{
  double c = 1.0 - x0;      // (1 - mu) = c - u
  double d = 1.0 - x0 * x0; // (1 - mu^2) = d - 2*x0*u - u^2

  SegmentMoments s;
  s.norm = a * u + 0.5 * m * u * u;
  s.m1 = a * c * u + 0.5 * (m * c - a) * u * u - m * u * u * u / 3.0;
  s.m2 =
    1.5 * (a * d * u + 0.5 * (m * d - 2.0 * a * x0) * u * u +
            (-2.0 * m * x0 - a) * u * u * u / 3.0 - 0.25 * m * u * u * u * u);
  return s;
}

//! Slope of the density on segment k
double segment_slope(
  const vector<double>& x, const vector<double>& p, bool histogram, int k)
{
  return histogram ? 0.0 : (p[k + 1] - p[k]) / (x[k + 1] - x[k]);
}

} // namespace

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

void AngleDistribution::transport_moments(
  vector<double>& energy, vector<double>& mu1, vector<double>& mu2) const
{
  energy = energy_;
  mu1.clear();
  mu2.clear();
  mu1.reserve(distribution_.size());
  mu2.reserve(distribution_.size());

  for (const auto& dist : distribution_) {
    double m1, m2;
    angular_moments(
      dist->x(), dist->p(), dist->interp() == Interpolation::histogram, m1, m2);
    mu1.push_back(m1);
    mu2.push_back(m2);
  }
}

void AngleDistribution::restricted_moments(double c1, vector<double>& energy,
  vector<double>& mu_cut, vector<double>& p_hard, vector<double>& mu1_soft,
  vector<double>& mu2_soft) const
{
  energy = energy_;
  mu_cut.clear();
  p_hard.clear();
  mu1_soft.clear();
  mu2_soft.clear();
  mu_cut.reserve(distribution_.size());
  p_hard.reserve(distribution_.size());
  mu1_soft.reserve(distribution_.size());
  mu2_soft.reserve(distribution_.size());

  for (const auto& dist : distribution_) {
    double cut, hard, m1, m2;
    restricted_angular_moments(dist->x(), dist->p(),
      dist->interp() == Interpolation::histogram, c1, cut, hard, m1, m2);
    mu_cut.push_back(cut);
    p_hard.push_back(hard);
    mu1_soft.push_back(m1);
    mu2_soft.push_back(m2);
  }
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

//==============================================================================
// Non-member functions
//==============================================================================

void angular_moments(const vector<double>& x, const vector<double>& p,
  bool histogram, double& mu1, double& mu2)
{
  int n = x.size();
  double norm = 0.0;
  double m1 = 0.0;
  double m2 = 0.0;
  for (int k = 0; k + 1 < n; ++k) {
    double h = x[k + 1] - x[k];
    if (h <= 0.0)
      continue;
    SegmentMoments seg =
      segment_moments(x[k], p[k], segment_slope(x, p, histogram, k), h);
    norm += seg.norm;
    m1 += seg.m1;
    m2 += seg.m2;
  }

  if (norm > 0.0) {
    mu1 = m1 / norm;
    mu2 = m2 / norm;
  } else {
    mu1 = 0.0;
    mu2 = 0.0;
  }
}

void restricted_angular_moments(const vector<double>& x,
  const vector<double>& p, bool histogram, double c1, double& mu_cut,
  double& p_hard, double& mu1_soft, double& mu2_soft)
{
  int n = x.size();

  // The single-event limit, which is also what a degenerate table gets: the
  // cutoff sits at the top of the range, so nothing is soft.
  mu_cut = n > 0 ? x[n - 1] : 1.0;
  p_hard = 1.0;
  mu1_soft = 0.0;
  mu2_soft = 0.0;
  if (n < 2 || c1 <= 0.0)
    return;

  // Running integral of the density from the bottom of the range. The hard
  // part is read from this rather than by subtracting the soft part from the
  // total: for the elastic distributions it is a ten-thousandth of the total,
  // and the difference of two nearly equal numbers would throw away most of
  // the precision in the one quantity that sets the step length.
  vector<double> prefix(n, 0.0);
  for (int k = 0; k + 1 < n; ++k) {
    double h = x[k + 1] - x[k];
    SegmentMoments seg = h > 0.0 ? segment_moments(x[k], p[k],
                                     segment_slope(x, p, histogram, k), h)
                                 : SegmentMoments {0.0, 0.0, 0.0};
    prefix[k + 1] = prefix[k] + seg.norm;
  }
  double norm = prefix[n - 1];
  if (norm <= 0.0)
    return;

  // Walk the cutoff down from mu = 1. The soft first moment rises and the hard
  // fraction falls, so their ratio increases monotonically; stop in the
  // segment where it first reaches C1.
  double soft_m1 = 0.0;
  double soft_m2 = 0.0;
  for (int k = n - 2; k >= 0; --k) {
    double x0 = x[k];
    double h = x[k + 1] - x0;
    if (h <= 0.0)
      continue;
    double a = p[k];
    double m = segment_slope(x, p, histogram, k);
    SegmentMoments full = segment_moments(x0, a, m, h);

    // Test the bottom of the segment, where the soft part is largest. A hard
    // fraction of zero there means the root is inside no matter what C1 is.
    if (prefix[k] <= 0.0 || soft_m1 + full.m1 >= c1 * prefix[k]) {
      // Bisect for the cutoff. The condition holds at u = 0 and fails at
      // u = h, the latter because the cutoff was still too high one segment
      // up. Sixty halvings take the bracket below the precision of a double.
      double ua = 0.0;
      double ub = h;
      for (int it = 0; it < 60; ++it) {
        double u = 0.5 * (ua + ub);
        SegmentMoments part = segment_moments(x0, a, m, u);
        double hard = prefix[k] + part.norm;
        double m1 = soft_m1 + full.m1 - part.m1;
        if (hard <= 0.0 || m1 >= c1 * hard) {
          ua = u;
        } else {
          ub = u;
        }
      }

      double u = 0.5 * (ua + ub);
      SegmentMoments part = segment_moments(x0, a, m, u);
      mu_cut = x0 + u;
      p_hard = (prefix[k] + part.norm) / norm;
      mu1_soft = (soft_m1 + full.m1 - part.m1) / norm;
      mu2_soft = (soft_m2 + full.m2 - part.m2) / norm;
      return;
    }

    soft_m1 += full.m1;
    soft_m2 += full.m2;
  }

  // Unreachable for a density that integrates to its own total, since the hard
  // fraction vanishes at the bottom of the range. Left in so that a table that
  // does not returns the condensed-history limit rather than nothing: every
  // collision soft, and no hard collision to end the step.
  mu_cut = x[0];
  p_hard = 0.0;
  mu1_soft = soft_m1 / norm;
  mu2_soft = soft_m2 / norm;
}

} // namespace openmc
