#include "openmc/scattdata.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include "openmc/tensor.h"

#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"

namespace openmc {

//==============================================================================
// ScattData base-class methods
//==============================================================================

void ScattData::base_init(int order, const tensor::Tensor<int>& in_gmin,
  const tensor::Tensor<int>& in_gmax, const double_2dvec& in_energy,
  const double_2dvec& in_mult)
{
  size_t groups = in_energy.size();

  gmin = in_gmin;
  gmax = in_gmax;
  energy.resize(groups);
  mult.resize(groups);
  dist.resize(groups);

  for (int gin = 0; gin < groups; gin++) {
    // Store the inputted data
    energy[gin] = in_energy[gin];
    mult[gin] = in_mult[gin];

    // Make sure the multiplicity does not have 0s
    for (int go = 0; go < mult[gin].size(); go++) {
      if (mult[gin][go] == 0.) {
        mult[gin][go] = 1.;
      }
    }

    // Make sure the energy is normalized
    double norm = std::accumulate(energy[gin].begin(), energy[gin].end(), 0.);

    if (norm != 0.) {
      for (auto& n : energy[gin])
        n /= norm;
    }

    // Initialize the distribution data
    dist[gin].resize(in_gmax[gin] - in_gmin[gin] + 1);
    for (auto& v : dist[gin]) {
      v.resize(order);
    }
  }
}

//==============================================================================

void ScattData::base_combine(size_t max_order, size_t order_dim,
  const vector<ScattData*>& those_scatts, const vector<double>& scalars,
  tensor::Tensor<int>& in_gmin, tensor::Tensor<int>& in_gmax,
  double_2dvec& sparse_mult, double_3dvec& sparse_scatter)
{
  size_t groups = those_scatts[0]->energy.size();

  // Now allocate and zero our storage spaces
  tensor::Tensor<double> this_nuscatt_matrix =
    tensor::zeros<double>({groups, groups, order_dim});
  tensor::Tensor<double> this_nuscatt_P0 =
    tensor::zeros<double>({groups, groups});
  tensor::Tensor<double> this_scatt_P0 =
    tensor::zeros<double>({groups, groups});
  tensor::Tensor<double> this_mult = tensor::ones<double>({groups, groups});

  // Build the dense scattering and multiplicity matrices
  for (int i = 0; i < those_scatts.size(); i++) {
    ScattData* that = those_scatts[i];

    // Build the dense matrix for that object
    tensor::Tensor<double> that_matrix = that->get_matrix(max_order);

    // Now add that to this for the nu-scatter matrix
    this_nuscatt_matrix += scalars[i] * that_matrix;

    // Do the same with the P0 matrices
    for (int gin = 0; gin < groups; gin++) {
      for (int go = 0; go < groups; go++) {
        this_nuscatt_P0(gin, go) +=
          scalars[i] * that->get_xs(MgxsType::NU_SCATTER, gin, &go, nullptr);
        this_scatt_P0(gin, go) +=
          scalars[i] * that->get_xs(MgxsType::SCATTER, gin, &go, nullptr);
      }
    }
  }

  // Now we have the dense nuscatt and scatt, we can easily compute the
  // multiplicity matrix by dividing the two and fixing any nans
  this_mult = tensor::nan_to_num(this_nuscatt_P0 / this_scatt_P0);

  // We have the data, now we need to convert to a jagged array and then use
  // the initialize function to store it on the object.
  for (int gin = 0; gin < groups; gin++) {
    // Find the minimum and maximum group boundaries
    int gmin_;
    for (gmin_ = 0; gmin_ < groups; gmin_++) {
      bool non_zero = false;
      for (int l = 0; l < this_nuscatt_matrix.shape(2); l++) {
        if (this_nuscatt_matrix(gin, gmin_, l) != 0.) {
          non_zero = true;
          break;
        }
      }
      if (non_zero)
        break;
    }
    int gmax_;
    for (gmax_ = groups - 1; gmax_ >= 0; gmax_--) {
      bool non_zero = false;
      for (int l = 0; l < this_nuscatt_matrix.shape(2); l++) {
        if (this_nuscatt_matrix(gin, gmax_, l) != 0.) {
          non_zero = true;
          break;
        }
      }
      if (non_zero)
        break;
    }

    // treat the case of all values being 0
    if (gmin_ > gmax_) {
      gmin_ = gin;
      gmax_ = gin;
    }

    // Store the group bounds
    in_gmin[gin] = gmin_;
    in_gmax[gin] = gmax_;

    // Store the data in the compressed format
    sparse_scatter[gin].resize(gmax_ - gmin_ + 1);
    sparse_mult[gin].resize(gmax_ - gmin_ + 1);
    int i_gout = 0;
    for (int gout = gmin_; gout <= gmax_; gout++) {
      sparse_scatter[gin][i_gout].resize(this_nuscatt_matrix.shape(2));
      for (int l = 0; l < this_nuscatt_matrix.shape(2); l++) {
        sparse_scatter[gin][i_gout][l] = this_nuscatt_matrix(gin, gout, l);
      }
      sparse_mult[gin][i_gout] = this_mult(gin, gout);
      i_gout++;
    }
  }
}

//==============================================================================

void ScattData::sample_energy(int gin, int& gout, int& i_gout, uint64_t* seed)
{
  // Sample the outgoing group
  double xi = prn(seed);
  double prob = 0.;
  i_gout = 0;
  for (gout = gmin[gin]; gout < gmax[gin]; ++gout) {
    prob += energy[gin][i_gout];
    if (xi < prob)
      break;
    ++i_gout;
  }
}

//==============================================================================

double ScattData::get_xs(
  MgxsType xstype, int gin, const int* gout, const double* mu)
{
  // Set the outgoing group offset index as needed
  int i_gout = 0;
  if (gout != nullptr) {
    // short circuit the function if gout is from a zero portion of the
    // scattering matrix
    if ((*gout < gmin[gin]) || (*gout > gmax[gin])) { // > gmax?
      return 0.;
    }
    i_gout = *gout - gmin[gin];
  }

  double val = scattxs[gin];
  switch (xstype) {
  case MgxsType::NU_SCATTER:
    if (gout != nullptr)
      val *= energy[gin][i_gout];
    break;
  case MgxsType::SCATTER:
    if (gout != nullptr) {
      val *= energy[gin][i_gout] / mult[gin][i_gout];
    } else {
      val /= std::inner_product(
        mult[gin].begin(), mult[gin].end(), energy[gin].begin(), 0.0);
    }
    break;
  case MgxsType::NU_SCATTER_FMU:
    if ((gout != nullptr) && (mu != nullptr)) {
      val *= energy[gin][i_gout] * calc_f(gin, *gout, *mu);
    } else {
      // This is not an expected path (asking for f_mu without asking for a
      // group or mu is not useful
      fatal_error("Invalid call to get_xs");
    }
    break;
  case MgxsType::SCATTER_FMU:
    if ((gout != nullptr) && (mu != nullptr)) {
      val *= energy[gin][i_gout] * calc_f(gin, *gout, *mu) / mult[gin][i_gout];
    } else {
      // This is not an expected path (asking for f_mu without asking for a
      // group or mu is not useful
      fatal_error("Invalid call to get_xs");
    }
    break;
  default:
    break;
  }
  return val;
}

//==============================================================================
// LegendreBetaMixture methods
//==============================================================================

namespace {

// Maximum number of bisections of [-1, 1] used to isolate points where a
// Legendre expansion is zero or changes sign
constexpr int MAX_SUBDIVISION_DEPTH {16};

// Binomial coefficients C(i, j) for 0 <= j <= i <= n
vector<vector<double>> binomial_table(int n)
{
  vector<vector<double>> binom(n + 1);
  for (int i = 0; i <= n; ++i) {
    binom[i].resize(i + 1);
    binom[i][0] = 1.;
    binom[i][i] = 1.;
    for (int j = 1; j < i; ++j) {
      binom[i][j] = binom[i - 1][j - 1] + binom[i - 1][j];
    }
  }
  return binom;
}

// An interval of t = (mu + 1)/2 and the Bernstein coefficients on it
struct BernsteinInterval {
  double lower;
  double width;
  int depth;
  vector<double> b;

  // Integral of the polynomial over the interval
  double mass() const
  {
    return width * std::accumulate(b.begin(), b.end(), 0.) / b.size();
  }

  // Integral of the polynomial above the smallest coefficient
  double residual() const
  {
    return mass() - width * *std::min_element(b.begin(), b.end());
  }

  // Value of the polynomial at the local coordinate u in [0, 1]
  double evaluate(double u) const
  {
    vector<double> w = b;
    for (int r = w.size() - 1; r > 0; --r) {
      for (int i = 0; i < r; ++i) {
        w[i] = (1. - u) * w[i] + u * w[i + 1];
      }
    }
    return w[0];
  }

  // Split the interval at the local coordinate u with the de Casteljau
  // algorithm
  std::pair<BernsteinInterval, BernsteinInterval> split(double u) const
  {
    int n = b.size() - 1;
    vector<double> left(n + 1);
    vector<double> right(n + 1);
    vector<double> w = b;
    for (int r = 0; r <= n; ++r) {
      left[r] = w[0];
      right[n - r] = w[n - r];
      for (int i = 0; i < n - r; ++i) {
        w[i] = (1. - u) * w[i] + u * w[i + 1];
      }
    }
    double x = lower + u * width;
    return {{lower, x - lower, depth + 1, std::move(left)},
      {x, lower + width - x, depth + 1, std::move(right)}};
  }

  std::pair<BernsteinInterval, BernsteinInterval> bisect() const
  {
    return split(0.5);
  }

  // If the polynomial changes sign exactly once on the interval, return the
  // local coordinate of its root; otherwise return a negative value. By the
  // variation diminishing property of the Bernstein basis, the number of roots
  // is at most the number of sign changes in the coefficients.
  double single_root() const
  {
    if (b.front() == 0. || b.back() == 0. ||
        (b.front() > 0.) == (b.back() > 0.))
      return -1.;
    int changes = 0;
    double last = b.front();
    for (double x : b) {
      if (x != 0. && (x > 0.) != (last > 0.)) {
        ++changes;
        last = x;
      }
    }
    if (changes != 1)
      return -1.;

    // Bisection on the polynomial
    double lo = 0.;
    double hi = 1.;
    bool positive_at_lo = b.front() > 0.;
    for (int i = 0; i < 60 && hi - lo > 1.e-15; ++i) {
      double mid = 0.5 * (lo + hi);
      if ((evaluate(mid) > 0.) == positive_at_lo) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return 0.5 * (lo + hi);
  }
};

// Build an alias table (Vose, DOI: 10.1109/32.92917) for sampling an index
// with probability proportional to p
void build_alias(
  const vector<double>& p, vector<double>& prob, vector<size_t>& alias)
{
  size_t n = p.size();
  double total = std::accumulate(p.begin(), p.end(), 0.);
  prob.resize(n);
  alias.assign(n, 0);
  vector<size_t> small;
  vector<size_t> large;
  for (size_t i = 0; i < n; ++i) {
    prob[i] = p[i] * n / total;
    if (prob[i] < 1.) {
      small.push_back(i);
    } else {
      large.push_back(i);
    }
  }
  while (!small.empty() && !large.empty()) {
    size_t j = small.back();
    small.pop_back();
    size_t k = large.back();
    alias[j] = k;
    prob[k] -= 1. - prob[j];
    if (prob[k] < 1.) {
      large.pop_back();
      small.push_back(k);
    }
  }
  // Entries left over are equal to one up to roundoff
  for (size_t i : large) {
    prob[i] = 1.;
  }
  for (size_t i : small) {
    prob[i] = 1.;
  }
}

// Sample an index from an alias table using a single uniform variate xi in
// [0, 1). The remaining randomness is returned in xi as a new uniform variate
// in [0, 1) that is independent of the index.
size_t sample_alias(
  const vector<double>& prob, const vector<size_t>& alias, double& xi)
{
  double x = xi * prob.size();
  size_t i = std::min(static_cast<size_t>(x), prob.size() - 1);
  double f = x - i;
  if (f < prob[i]) {
    xi = f / prob[i];
    return i;
  } else {
    xi = (f - prob[i]) / (1. - prob[i]);
    return alias[i];
  }
}

} // namespace

LegendreBetaMixture::LegendreBetaMixture(
  const vector<double>& coeffs, double max_residual, int max_bisections)
{
  int n = coeffs.size() - 1;
  auto binom = binomial_table(n);

  // Bernstein coefficients on [0, 1] of the density in t = (mu + 1)/2, which
  // is g(t) = sum_l (2l + 1) a_l P_l(2t - 1). They follow from the identity
  // P_l(2t - 1) = sum_j (-1)^(l-j) C(l,j)^2 t^j (1-t)^(l-j) after elevating
  // the degree from l to n.
  vector<double> b(n + 1, 0.);
  for (int k = 0; k <= n; ++k) {
    for (int l = 0; l <= n; ++l) {
      double sum = 0.;
      for (int j = std::max(0, k - n + l); j <= std::min(l, k); ++j) {
        double term = binom[l][j] * binom[l][j] * binom[n - l][k - j];
        sum += ((l - j) % 2 == 0) ? term : -term;
      }
      b[k] += (2 * l + 1) * coeffs[l] * sum;
    }
    b[k] /= binom[n][k];
  }

  // Coefficients within roundoff of zero are treated as zero
  double tolerance = 0.;
  for (double x : b) {
    tolerance = std::max(tolerance, std::abs(x));
  }
  tolerance *= 1.e-12;

  // Bisect [0, 1] until the coefficients on each interval are nonnegative
  vector<BernsteinInterval> intervals;
  vector<BernsteinInterval> stack {{0., 1., 0, b}};
  while (!stack.empty()) {
    BernsteinInterval interval = std::move(stack.back());
    stack.pop_back();
    auto [bmin, bmax] =
      std::minmax_element(interval.b.begin(), interval.b.end());

    // The expansion is not positive on this interval
    if (*bmax <= tolerance)
      continue;

    if (*bmin >= -tolerance || interval.depth == MAX_SUBDIVISION_DEPTH) {
      // Keep the interval. At the maximum depth it contains a point where the
      // expansion is zero or changes sign, and only its positive part is kept.
      for (auto& x : interval.b) {
        x = std::max(x, 0.);
      }
      intervals.push_back(std::move(interval));
      continue;
    }

    // Split at the root if there is exactly one, and otherwise in half
    double root = interval.single_root();
    auto [left, right] =
      root > 0. && root < 1. ? interval.split(root) : interval.bisect();
    stack.push_back(std::move(right));
    stack.push_back(std::move(left));
  }

  double total = 0.;
  for (const auto& interval : intervals) {
    total += interval.mass();
  }
  if (intervals.empty() || total <= 0.) {
    // There is nothing to sample, so fall back to an isotropic distribution
    intervals = {{0., 1., 0, {1.}}};
    total = 1.;
  }

  // Bisect the interval with the largest probability above its floor until
  // the histogram of floors carries enough of the probability
  for (int i_bisection = 0; i_bisection < max_bisections; ++i_bisection) {
    double residual = 0.;
    size_t worst = 0;
    for (size_t i = 0; i < intervals.size(); ++i) {
      double r = intervals[i].residual();
      residual += r;
      if (r > intervals[worst].residual())
        worst = i;
    }
    if (residual <= max_residual * total)
      break;
    auto [left, right] = intervals[worst].bisect();
    intervals[worst] = std::move(left);
    intervals.insert(intervals.begin() + worst + 1, std::move(right));
  }

  // Store the intervals in mu with coefficients of the normalized density
  order_ = intervals[0].b.size() - 1;
  vector<double> hist_weight;
  vector<double> resid_weight;
  for (const auto& interval : intervals) {
    lower_.push_back(2. * interval.lower - 1.);
    width_.push_back(2. * interval.width);
    double floor =
      *std::min_element(interval.b.begin(), interval.b.end()) / total;
    double sum = 0.;
    for (double x : interval.b) {
      coeffs_.push_back(x / total);
      sum += x / total - floor;
    }
    floor_.push_back(floor);
    residual_sum_.push_back(sum);
    hist_weight.push_back(interval.width * floor);
    resid_weight.push_back(interval.width * sum / (order_ + 1));
  }

  double hist_total =
    std::accumulate(hist_weight.begin(), hist_weight.end(), 0.);
  double resid_total =
    std::accumulate(resid_weight.begin(), resid_weight.end(), 0.);
  histogram_fraction_ = hist_total / (hist_total + resid_total);
  if (hist_total > 0.) {
    inv_histogram_fraction_ = 1. / histogram_fraction_;
    build_alias(hist_weight, hist_prob_, hist_alias_);

    // Precompute the factors that map the randomness left over from the alias
    // decision to a position within the selected interval
    size_t n_slots = hist_prob_.size();
    hist_scale_.resize(n_slots);
    hist_alias_scale_.resize(n_slots);
    for (size_t i = 0; i < n_slots; ++i) {
      double p = hist_prob_[i];
      hist_scale_[i] = p > 0. ? width_[i] / p : 0.;
      hist_alias_scale_[i] = p < 1. ? width_[hist_alias_[i]] / (1. - p) : 0.;
    }
  }
  if (resid_total > 0.)
    build_alias(resid_weight, resid_prob_, resid_alias_);
}

double LegendreBetaMixture::sample(uint64_t* seed) const
{
  double xi = prn(seed);
  if (xi < histogram_fraction_) {
    // Sample the histogram of floors: pick an interval with the alias method
    // and place mu uniformly within it, all from the same random number. The
    // fraction left over after choosing a slot of the alias table is uniform
    // on [0, prob) for the slot itself and on [prob, 1) for its alias.
    double x = xi * inv_histogram_fraction_ * hist_prob_.size();
    size_t i = std::min(static_cast<size_t>(x), hist_prob_.size() - 1);
    double f = x - i;
    double mu;
    if (f < hist_prob_[i]) {
      mu = lower_[i] + f * hist_scale_[i];
    } else {
      size_t j = hist_alias_[i];
      mu = lower_[j] + (f - hist_prob_[i]) * hist_alias_scale_[i];
    }
    return std::clamp(mu, -1., 1.);
  }

  // Sample the residual: pick an interval, then a beta component in it with
  // probability proportional to its coefficient above the floor
  xi = (xi - histogram_fraction_) / (1. - histogram_fraction_);
  size_t j = sample_alias(resid_prob_, resid_alias_, xi);
  const double* c = &coeffs_[j * (order_ + 1)];
  double target = xi * residual_sum_[j];
  int k = 0;
  for (; k < order_; ++k) {
    target -= c[k] - floor_[j];
    if (target < 0.)
      break;
  }
  double u = beta_variate(k + 1, order_ - k + 1, seed);
  return std::clamp(lower_[j] + width_[j] * u, -1., 1.);
}

double LegendreBetaMixture::evaluate(double mu) const
{
  // Find the interval containing mu; the last one includes its upper bound
  int n_intervals = lower_.size();
  int j = std::upper_bound(lower_.begin(), lower_.end(), mu) - lower_.begin();
  if (j == 0 || mu > 1.)
    return 0.;
  --j;
  double u = (mu - lower_[j]) / width_[j];
  if (u > 1. || (u == 1. && j < n_intervals - 1))
    return 0.;

  // Evaluate the polynomial in Bernstein form; the factor of 1/2 converts the
  // density in t to a density in mu
  int n = order_;
  const double* c = &coeffs_[j * (n + 1)];
  double f = 0.;
  double binom = 1.;
  for (int k = 0; k <= n; ++k) {
    f += c[k] * binom * std::pow(u, k) * std::pow(1. - u, n - k);
    binom = binom * (n - k) / (k + 1);
  }
  return 0.5 * f;
}

//==============================================================================
// ScattDataLegendre methods
//==============================================================================

void ScattDataLegendre::init(const tensor::Tensor<int>& in_gmin,
  const tensor::Tensor<int>& in_gmax, const double_2dvec& in_mult,
  const double_3dvec& coeffs)
{
  size_t groups = coeffs.size();
  size_t order = coeffs[0][0].size();

  // make a copy of coeffs that we can use to both extract data and normalize
  double_3dvec matrix = coeffs;

  // Get the scattering cross section value by summing the un-normalized P0
  // coefficient in the variable matrix over all outgoing groups.
  scattxs = tensor::zeros<double>({groups});
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = in_gmax[gin] - in_gmin[gin] + 1;
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      scattxs[gin] += matrix[gin][i_gout][0];
    }
  }

  // Build the energy transfer matrix from data in the variable matrix while
  // also normalizing the variable matrix itself
  // (forcing the CDF of f(mu=1) == 1)
  double_2dvec in_energy;
  in_energy.resize(groups);
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = in_gmax[gin] - in_gmin[gin] + 1;
    in_energy[gin].resize(num_groups);
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      double norm = matrix[gin][i_gout][0];
      in_energy[gin][i_gout] = norm;
      if (norm != 0.) {
        for (auto& n : matrix[gin][i_gout])
          n /= norm;
      }
    }
  }

  // Initialize the base class attributes
  ScattData::base_init(order, in_gmin, in_gmax, in_energy, in_mult);

  // Set the distribution (sdata.dist) values. The representation used for
  // sampling is only built by combine(), since only the combined data of a
  // material is sampled.
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = gmax[gin] - gmin[gin] + 1;
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      dist[gin][i_gout] = matrix[gin][i_gout];
    }
  }
  mixture.clear();
}

//==============================================================================

void ScattDataLegendre::update_mixture()
{
  size_t groups = dist.size();
  mixture.resize(groups);
  for (int gin = 0; gin < groups; gin++) {
    mixture[gin].clear();
    for (const auto& coeffs : dist[gin]) {
      mixture[gin].emplace_back(coeffs);
    }
  }
}

//==============================================================================

double ScattDataLegendre::calc_f(int gin, int gout, double mu)
{
  double f;
  if ((gout < gmin[gin]) || (gout > gmax[gin])) {
    f = 0.;
  } else {
    int i_gout = gout - gmin[gin];
    f = evaluate_legendre(
      dist[gin][i_gout].size() - 1, dist[gin][i_gout].data(), mu);
  }
  return f;
}

//==============================================================================

void ScattDataLegendre::sample(
  int gin, int& gout, double& mu, double& wgt, uint64_t* seed)
{
  // Sample the outgoing energy using the base-class method
  int i_gout;
  sample_energy(gin, gout, i_gout, seed);

  // Sample mu directly from the beta mixture representation of the
  // scattering kernel
  mu = mixture[gin][i_gout].sample(seed);

  // Update the weight to reflect neutron multiplicity
  wgt *= mult[gin][i_gout];
}

//==============================================================================

void ScattDataLegendre::combine(
  const vector<ScattData*>& those_scatts, const vector<double>& scalars)
{
  // Find the max order in the data set and make sure we can combine the sets
  size_t max_order = 0;
  for (int i = 0; i < those_scatts.size(); i++) {
    // Lets also make sure these items are combineable
    ScattDataLegendre* that = dynamic_cast<ScattDataLegendre*>(those_scatts[i]);
    if (!that) {
      fatal_error("Cannot combine the ScattData objects!");
    }
    size_t that_order = that->get_order();
    if (that_order > max_order)
      max_order = that_order;
  }

  size_t groups = those_scatts[0]->energy.size();

  tensor::Tensor<int> in_gmin({groups}, 0);
  tensor::Tensor<int> in_gmax({groups}, 0);
  double_3dvec sparse_scatter(groups);
  double_2dvec sparse_mult(groups);

  // The rest of the steps do not depend on the type of angular representation
  // so we use a base class method to sum up xs and create new energy and mult
  // matrices
  size_t order_dim = max_order + 1;
  ScattData::base_combine(max_order, order_dim, those_scatts, scalars, in_gmin,
    in_gmax, sparse_mult, sparse_scatter);

  // Got everything we need, store it.
  init(in_gmin, in_gmax, sparse_mult, sparse_scatter);

  // Build the representation used for sampling
  update_mixture();
}

//==============================================================================

tensor::Tensor<double> ScattDataLegendre::get_matrix(size_t max_order)
{
  // Get the sizes and initialize the data to 0
  size_t groups = energy.size();
  size_t order_dim = max_order + 1;
  tensor::Tensor<double> matrix =
    tensor::zeros<double>({groups, groups, order_dim});

  for (int gin = 0; gin < groups; gin++) {
    for (int i_gout = 0; i_gout < energy[gin].size(); i_gout++) {
      int gout = i_gout + gmin[gin];
      for (int l = 0; l < order_dim; l++) {
        matrix(gin, gout, l) =
          scattxs[gin] * energy[gin][i_gout] * dist[gin][i_gout][l];
      }
    }
  }
  return matrix;
}

//==============================================================================
// ScattDataHistogram methods
//==============================================================================

void ScattDataHistogram::init(const tensor::Tensor<int>& in_gmin,
  const tensor::Tensor<int>& in_gmax, const double_2dvec& in_mult,
  const double_3dvec& coeffs)
{
  size_t groups = coeffs.size();
  size_t order = coeffs[0][0].size();

  // make a copy of coeffs that we can use to both extract data and normalize
  double_3dvec matrix = coeffs;

  // Get the scattering cross section value by summing the distribution
  // over all the histogram bins in angle and outgoing energy groups
  scattxs = tensor::zeros<double>({groups});
  for (int gin = 0; gin < groups; gin++) {
    for (int i_gout = 0; i_gout < matrix[gin].size(); i_gout++) {
      scattxs[gin] += std::accumulate(
        matrix[gin][i_gout].begin(), matrix[gin][i_gout].end(), 0.);
    }
  }

  // Build the energy transfer matrix from data in the variable matrix
  double_2dvec in_energy;
  in_energy.resize(groups);
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = in_gmax[gin] - in_gmin[gin] + 1;
    in_energy[gin].resize(num_groups);
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      double norm = std::accumulate(
        matrix[gin][i_gout].begin(), matrix[gin][i_gout].end(), 0.);
      in_energy[gin][i_gout] = norm;
      if (norm != 0.) {
        for (auto& n : matrix[gin][i_gout])
          n /= norm;
      }
    }
  }

  // Initialize the base class attributes
  ScattData::base_init(order, in_gmin, in_gmax, in_energy, in_mult);

  // Build the angular distribution mu values
  mu = tensor::linspace(-1., 1., order + 1);
  dmu = 2. / order;

  // Calculate f(mu) and integrate it so we can avoid rejection sampling
  fmu.resize(groups);
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = gmax[gin] - gmin[gin] + 1;
    fmu[gin].resize(num_groups);
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      fmu[gin][i_gout].resize(order);
      // The variable matrix contains f(mu); so directly assign it
      fmu[gin][i_gout] = matrix[gin][i_gout];

      // Integrate the histogram
      dist[gin][i_gout][0] = dmu * matrix[gin][i_gout][0];
      for (int imu = 1; imu < order; imu++) {
        dist[gin][i_gout][imu] =
          dmu * matrix[gin][i_gout][imu] + dist[gin][i_gout][imu - 1];
      }

      // Now re-normalize for integral to unity
      double norm = dist[gin][i_gout][order - 1];
      if (norm > 0.) {
        for (int imu = 0; imu < order; imu++) {
          fmu[gin][i_gout][imu] /= norm;
          dist[gin][i_gout][imu] /= norm;
        }
      }
    }
  }
}

//==============================================================================

double ScattDataHistogram::calc_f(int gin, int gout, double mu)
{
  double f;
  if ((gout < gmin[gin]) || (gout > gmax[gin])) {
    f = 0.;
  } else {
    // Find mu bin
    int i_gout = gout - gmin[gin];
    int imu;
    if (mu == 1.) {
      // use size -2 to have the index one before the end
      imu = this->mu.shape(0) - 2;
    } else {
      imu = std::floor((mu + 1.) / dmu + 1.) - 1;
    }

    f = fmu[gin][i_gout][imu];
  }
  return f;
}

//==============================================================================

void ScattDataHistogram::sample(
  int gin, int& gout, double& mu, double& wgt, uint64_t* seed)
{
  // Sample the outgoing energy using the base-class method
  int i_gout;
  sample_energy(gin, gout, i_gout, seed);

  // Determine the outgoing cosine bin
  double xi = prn(seed);

  int imu;
  if (xi < dist[gin][i_gout][0]) {
    imu = 0;
  } else {
    imu =
      std::upper_bound(dist[gin][i_gout].begin(), dist[gin][i_gout].end(), xi) -
      dist[gin][i_gout].begin();
  }

  // Randomly select mu within the imu bin
  mu = prn(seed) * dmu + this->mu[imu];

  mu = std::clamp(mu, -1., 1.);

  // Update the weight to reflect neutron multiplicity
  wgt *= mult[gin][i_gout];
}

//==============================================================================

tensor::Tensor<double> ScattDataHistogram::get_matrix(size_t max_order)
{
  // Get the sizes and initialize the data to 0
  size_t groups = energy.size();
  // We ignore the requested order for Histogram and Tabular representations
  size_t order_dim = get_order();
  tensor::Tensor<double> matrix({groups, groups, order_dim}, 0);

  for (int gin = 0; gin < groups; gin++) {
    for (int i_gout = 0; i_gout < energy[gin].size(); i_gout++) {
      int gout = i_gout + gmin[gin];
      for (int l = 0; l < order_dim; l++) {
        matrix(gin, gout, l) =
          scattxs[gin] * energy[gin][i_gout] * fmu[gin][i_gout][l];
      }
    }
  }
  return matrix;
}

//==============================================================================

void ScattDataHistogram::combine(
  const vector<ScattData*>& those_scatts, const vector<double>& scalars)
{
  // Find the max order in the data set and make sure we can combine the sets
  size_t max_order = those_scatts[0]->get_order();
  for (int i = 0; i < those_scatts.size(); i++) {
    // Lets also make sure these items are combineable
    ScattDataHistogram* that =
      dynamic_cast<ScattDataHistogram*>(those_scatts[i]);
    if (!that) {
      fatal_error("Cannot combine the ScattData objects!");
    }
    if (max_order != that->get_order()) {
      fatal_error("Cannot combine the ScattData objects!");
    }
  }

  size_t groups = those_scatts[0]->energy.size();

  tensor::Tensor<int> in_gmin({groups}, 0);
  tensor::Tensor<int> in_gmax({groups}, 0);
  double_3dvec sparse_scatter(groups);
  double_2dvec sparse_mult(groups);

  // The rest of the steps do not depend on the type of angular representation
  // so we use a base class method to sum up xs and create new energy and mult
  // matrices
  size_t order_dim = max_order;
  ScattData::base_combine(max_order, order_dim, those_scatts, scalars, in_gmin,
    in_gmax, sparse_mult, sparse_scatter);

  // Got everything we need, store it.
  init(in_gmin, in_gmax, sparse_mult, sparse_scatter);
}

//==============================================================================
// ScattDataTabular methods
//==============================================================================

void ScattDataTabular::init(const tensor::Tensor<int>& in_gmin,
  const tensor::Tensor<int>& in_gmax, const double_2dvec& in_mult,
  const double_3dvec& coeffs)
{
  size_t groups = coeffs.size();
  size_t order = coeffs[0][0].size();

  // make a copy of coeffs that we can use to both extract data and normalize
  double_3dvec matrix = coeffs;

  // Build the angular distribution mu values
  mu = tensor::linspace(-1., 1., order);
  dmu = 2. / (order - 1);

  // Get the scattering cross section value by integrating the distribution
  // over all mu points and then combining over all outgoing groups
  scattxs = tensor::zeros<double>({groups});
  for (int gin = 0; gin < groups; gin++) {
    for (int i_gout = 0; i_gout < matrix[gin].size(); i_gout++) {
      for (int imu = 1; imu < order; imu++) {
        scattxs[gin] +=
          0.5 * dmu * (matrix[gin][i_gout][imu - 1] + matrix[gin][i_gout][imu]);
      }
    }
  }

  // Build the energy transfer matrix from data in the variable matrix
  double_2dvec in_energy(groups);
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = in_gmax[gin] - in_gmin[gin] + 1;
    in_energy[gin].resize(num_groups);
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      double norm = 0.;
      for (int imu = 1; imu < order; imu++) {
        norm +=
          0.5 * dmu * (matrix[gin][i_gout][imu - 1] + matrix[gin][i_gout][imu]);
      }
      in_energy[gin][i_gout] = norm;
    }
  }

  // Initialize the base class attributes
  ScattData::base_init(order, in_gmin, in_gmax, in_energy, in_mult);

  // Calculate f(mu) and integrate it so we can avoid rejection sampling
  fmu.resize(groups);
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = gmax[gin] - gmin[gin] + 1;
    fmu[gin].resize(num_groups);
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      fmu[gin][i_gout].resize(order);
      // The variable matrix contains f(mu); so directly assign it
      fmu[gin][i_gout] = matrix[gin][i_gout];

      // Ensure positivity
      for (auto& val : fmu[gin][i_gout]) {
        if (val < 0.)
          val = 0.;
      }

      // Now re-normalize for numerical integration issues and to take care of
      // the above negative fix-up.  Also accrue the CDF
      double norm = 0.;
      for (int imu = 1; imu < order; imu++) {
        norm += 0.5 * dmu * (fmu[gin][i_gout][imu - 1] + fmu[gin][i_gout][imu]);
        // incorporate to the CDF
        dist[gin][i_gout][imu] = norm;
      }

      // now do the normalization
      if (norm > 0.) {
        for (int imu = 0; imu < order; imu++) {
          fmu[gin][i_gout][imu] /= norm;
          dist[gin][i_gout][imu] /= norm;
        }
      }
    }
  }
}

//==============================================================================

double ScattDataTabular::calc_f(int gin, int gout, double mu)
{
  double f;
  if ((gout < gmin[gin]) || (gout > gmax[gin])) {
    f = 0.;
  } else {
    // Find mu bin
    int i_gout = gout - gmin[gin];
    int imu;
    if (mu == 1.) {
      // use size -2 to have the index one before the end
      imu = this->mu.shape(0) - 2;
    } else {
      imu = std::floor((mu + 1.) / dmu + 1.) - 1;
    }

    double r = (mu - this->mu[imu]) / (this->mu[imu + 1] - this->mu[imu]);
    f = (1. - r) * fmu[gin][i_gout][imu] + r * fmu[gin][i_gout][imu + 1];
  }
  return f;
}

//==============================================================================

void ScattDataTabular::sample(
  int gin, int& gout, double& mu, double& wgt, uint64_t* seed)
{
  // Sample the outgoing energy using the base-class method
  int i_gout;
  sample_energy(gin, gout, i_gout, seed);

  // Determine the outgoing cosine bin
  int NP = this->mu.shape(0);
  double xi = prn(seed);

  double c_k = dist[gin][i_gout][0];
  int k;
  for (k = 0; k < NP - 1; k++) {
    double c_k1 = dist[gin][i_gout][k + 1];
    if (xi < c_k1)
      break;
    c_k = c_k1;
  }

  // Check to make sure k is <= NP - 1
  k = std::min(k, NP - 2);

  // Find the pdf values we want
  double p0 = fmu[gin][i_gout][k];
  double mu0 = this->mu[k];
  double p1 = fmu[gin][i_gout][k + 1];
  double mu1 = this->mu[k + 1];

  if (p0 == p1) {
    mu = mu0 + (xi - c_k) / p0;
  } else {
    double frac = (p1 - p0) / (mu1 - mu0);
    mu =
      mu0 +
      (std::sqrt(std::max(0., p0 * p0 + 2. * frac * (xi - c_k))) - p0) / frac;
  }

  mu = std::clamp(mu, -1., 1.);

  // Update the weight to reflect neutron multiplicity
  wgt *= mult[gin][i_gout];
}

//==============================================================================

tensor::Tensor<double> ScattDataTabular::get_matrix(size_t max_order)
{
  // Get the sizes and initialize the data to 0
  size_t groups = energy.size();
  // We ignore the requested order for Histogram and Tabular representations
  size_t order_dim = get_order();
  tensor::Tensor<double> matrix =
    tensor::zeros<double>({groups, groups, order_dim});

  for (int gin = 0; gin < groups; gin++) {
    for (int i_gout = 0; i_gout < energy[gin].size(); i_gout++) {
      int gout = i_gout + gmin[gin];
      for (int l = 0; l < order_dim; l++) {
        matrix(gin, gout, l) =
          scattxs[gin] * energy[gin][i_gout] * fmu[gin][i_gout][l];
      }
    }
  }
  return matrix;
}

//==============================================================================

void ScattDataTabular::combine(
  const vector<ScattData*>& those_scatts, const vector<double>& scalars)
{
  // Find the max order in the data set and make sure we can combine the sets
  size_t max_order = those_scatts[0]->get_order();
  for (int i = 0; i < those_scatts.size(); i++) {
    // Lets also make sure these items are combineable
    ScattDataTabular* that = dynamic_cast<ScattDataTabular*>(those_scatts[i]);
    if (!that) {
      fatal_error("Cannot combine the ScattData objects!");
    }
    if (max_order != that->get_order()) {
      fatal_error("Cannot combine the ScattData objects!");
    }
  }

  size_t groups = those_scatts[0]->energy.size();

  tensor::Tensor<int> in_gmin({groups}, 0);
  tensor::Tensor<int> in_gmax({groups}, 0);
  double_3dvec sparse_scatter(groups);
  double_2dvec sparse_mult(groups);

  // The rest of the steps do not depend on the type of angular representation
  // so we use a base class method to sum up xs and create new energy and mult
  // matrices
  size_t order_dim = max_order;
  ScattData::base_combine(max_order, order_dim, those_scatts, scalars, in_gmin,
    in_gmax, sparse_mult, sparse_scatter);

  // Got everything we need, store it.
  init(in_gmin, in_gmax, sparse_mult, sparse_scatter);
}

//==============================================================================
// ScattDataNEPSF methods
//==============================================================================

void ScattDataNEPSF::init(const tensor::Tensor<int>& in_gmin,
  const tensor::Tensor<int>& in_gmax, const double_2dvec& in_mult,
  const double_3dvec& coeffs)
{
  size_t groups = coeffs.size();

  // The group-to-group cross section is the sum of the step weights
  scattxs = tensor::zeros<double>({groups});
  double_2dvec in_energy(groups);
  for (int gin = 0; gin < groups; gin++) {
    in_energy[gin].resize(coeffs[gin].size());
    for (int i_gout = 0; i_gout < coeffs[gin].size(); i_gout++) {
      const auto& c = coeffs[gin][i_gout];
      double norm = 0.;
      for (size_t k = 0; k < c.size(); k += 3) {
        norm += c[k];
      }
      in_energy[gin][i_gout] = norm;
      scattxs[gin] += norm;
    }
  }

  // Initialize the base class attributes
  ScattData::base_init(0, in_gmin, in_gmax, in_energy, in_mult);

  // Data from a library has the same number of steps for every transfer
  if (groups > 0 && !coeffs[0].empty() && coeffs[0][0].size() >= 3) {
    order = 2 * (coeffs[0][0].size() / 3) - 1;
  }

  steps.resize(groups);
  for (int gin = 0; gin < groups; gin++) {
    steps[gin].resize(coeffs[gin].size());
    for (int i_gout = 0; i_gout < coeffs[gin].size(); i_gout++) {
      const auto& c = coeffs[gin][i_gout];
      double norm = in_energy[gin][i_gout];
      Steps& s = steps[gin][i_gout];
      vector<double> weights;
      for (size_t k = 0; k + 2 < c.size(); k += 3) {
        if (c[k] == 0. || norm == 0.)
          continue;
        double p = c[k] / norm;
        dist[gin][i_gout].insert(
          dist[gin][i_gout].end(), {p, c[k + 1], c[k + 2]});
        s.lower.push_back(c[k + 1]);
        s.width.push_back(c[k + 2] - c[k + 1]);
        // Steps with negative weight, which only come from negative
        // group-to-group cross sections, are not sampled
        weights.push_back(std::max(p, 0.));
      }
      if (std::accumulate(weights.begin(), weights.end(), 0.) > 0.) {
        build_alias(weights, s.prob, s.alias);
      }
    }
  }
}

//==============================================================================

void ScattDataNEPSF::combine(
  const vector<ScattData*>& those_scatts, const vector<double>& scalars)
{
  size_t groups = those_scatts[0]->energy.size();

  // Concatenate the steps of each transfer with weights equal to their
  // contribution to the combined cross section
  vector<vector<vector<double>>> dense(groups, vector<vector<double>>(groups));
  tensor::Tensor<double> nuscatt = tensor::zeros<double>({groups, groups});
  tensor::Tensor<double> scatt = tensor::zeros<double>({groups, groups});
  size_t max_order = 0;
  for (int i = 0; i < those_scatts.size(); i++) {
    auto* that = dynamic_cast<ScattDataNEPSF*>(those_scatts[i]);
    if (!that) {
      fatal_error("Cannot combine the ScattData objects!");
    }
    max_order = std::max(max_order, that->get_order());
    for (int gin = 0; gin < groups; gin++) {
      for (int i_gout = 0; i_gout < that->energy[gin].size(); i_gout++) {
        int gout = i_gout + that->gmin[gin];
        double xs = scalars[i] * that->scattxs[gin] * that->energy[gin][i_gout];
        if (xs == 0.)
          continue;
        nuscatt(gin, gout) += xs;
        scatt(gin, gout) += xs / that->mult[gin][i_gout];
        const auto& d = that->dist[gin][i_gout];
        for (size_t k = 0; k < d.size(); k += 3) {
          dense[gin][gout].insert(
            dense[gin][gout].end(), {xs * d[k], d[k + 1], d[k + 2]});
        }
      }
    }
  }

  // Convert to the sparse format
  tensor::Tensor<int> in_gmin({groups}, 0);
  tensor::Tensor<int> in_gmax({groups}, 0);
  double_3dvec sparse_scatter(groups);
  double_2dvec sparse_mult(groups);
  for (int gin = 0; gin < groups; gin++) {
    int gmin_ = 0;
    while (gmin_ < groups && dense[gin][gmin_].empty())
      ++gmin_;
    int gmax_ = groups - 1;
    while (gmax_ >= 0 && dense[gin][gmax_].empty())
      --gmax_;
    if (gmin_ > gmax_) {
      gmin_ = gin;
      gmax_ = gin;
    }
    in_gmin[gin] = gmin_;
    in_gmax[gin] = gmax_;
    for (int gout = gmin_; gout <= gmax_; gout++) {
      sparse_scatter[gin].push_back(std::move(dense[gin][gout]));
      double m = nuscatt(gin, gout) / scatt(gin, gout);
      sparse_mult[gin].push_back(std::isfinite(m) ? m : 0.);
    }
  }

  init(in_gmin, in_gmax, sparse_mult, sparse_scatter);
  order = max_order;
}

//==============================================================================

double ScattDataNEPSF::calc_f(int gin, int gout, double mu)
{
  if ((gout < gmin[gin]) || (gout > gmax[gin])) {
    return 0.;
  }
  const auto& d = dist[gin][gout - gmin[gin]];
  double f = 0.;
  for (size_t k = 0; k < d.size(); k += 3) {
    double lower = d[k + 1];
    double upper = d[k + 2];
    if (upper > lower && mu >= lower && (mu < upper || mu == 1.)) {
      f += d[k] / (upper - lower);
    }
  }
  return f;
}

//==============================================================================

void ScattDataNEPSF::sample(
  int gin, int& gout, double& mu, double& wgt, uint64_t* seed)
{
  // Sample the outgoing energy using the base-class method
  int i_gout;
  sample_energy(gin, gout, i_gout, seed);

  const Steps& s = steps[gin][i_gout];
  double xi = prn(seed);
  if (s.prob.empty()) {
    // No angular data for this transfer
    mu = 2. * xi - 1.;
  } else {
    // Choose a step with the alias method, then a point within it from the
    // randomness left over
    size_t i = sample_alias(s.prob, s.alias, xi);
    mu = std::clamp(s.lower[i] + xi * s.width[i], -1., 1.);
  }

  // Update the weight to reflect neutron multiplicity
  wgt *= mult[gin][i_gout];
}

//==============================================================================

tensor::Tensor<double> ScattDataNEPSF::get_matrix(size_t max_order)
{
  size_t groups = energy.size();
  size_t order_dim = max_order + 1;
  tensor::Tensor<double> matrix =
    tensor::zeros<double>({groups, groups, order_dim});

  // Legendre polynomials up to order max_order + 1 at the step bounds
  vector<double> p_lower(order_dim + 1);
  vector<double> p_upper(order_dim + 1);
  vector<double> p_mid(order_dim + 1);
  for (int gin = 0; gin < groups; gin++) {
    for (int i_gout = 0; i_gout < energy[gin].size(); i_gout++) {
      int gout = i_gout + gmin[gin];
      double xs = scattxs[gin] * energy[gin][i_gout];
      const auto& d = dist[gin][i_gout];
      for (size_t k = 0; k < d.size(); k += 3) {
        double lower = d[k + 1];
        double upper = d[k + 2];
        double width = upper - lower;
        if (width < 1.e-6) {
          // The mean of P_l over a narrow step is P_l at its midpoint to
          // within O(width^2)
          calc_pn_c(max_order, 0.5 * (lower + upper), p_mid.data());
          for (int l = 0; l < order_dim; l++) {
            matrix(gin, gout, l) += xs * d[k] * p_mid[l];
          }
        } else {
          // The integral of P_l is (P_l+1 - P_l-1)/(2l + 1)
          calc_pn_c(max_order + 1, lower, p_lower.data());
          calc_pn_c(max_order + 1, upper, p_upper.data());
          matrix(gin, gout, 0) += xs * d[k];
          for (int l = 1; l < order_dim; l++) {
            double integral = (p_upper[l + 1] - p_upper[l - 1] -
                                p_lower[l + 1] + p_lower[l - 1]) /
                              (2 * l + 1);
            matrix(gin, gout, l) += xs * d[k] * integral / width;
          }
        }
      }
    }
  }
  return matrix;
}

//==============================================================================
// module-level methods
//==============================================================================

void convert_legendre_to_tabular(ScattDataLegendre& leg, ScattDataTabular& tab)
{
  // See if the user wants us to figure out how many points to use
  int n_mu = settings::legendre_to_tabular_points;
  if (n_mu == C_NONE) {
    // then we will use 2 pts if its P0, or the default if a higher order
    // TODO use an error minimization algorithm that also picks n_mu
    if (leg.get_order() == 0) {
      n_mu = 2;
    } else {
      n_mu = DEFAULT_NMU;
    }
  }

  tab.base_init(n_mu, leg.gmin, leg.gmax, leg.energy, leg.mult);
  tab.scattxs = leg.scattxs;

  // Build mu and dmu
  tab.mu = tensor::linspace(-1., 1., n_mu);
  tab.dmu = 2. / (n_mu - 1);

  // Calculate f(mu) and integrate it so we can avoid rejection sampling
  size_t groups = tab.energy.size();
  tab.fmu.resize(groups);
  for (int gin = 0; gin < groups; gin++) {
    int num_groups = tab.gmax[gin] - tab.gmin[gin] + 1;
    tab.fmu[gin].resize(num_groups);
    for (int i_gout = 0; i_gout < num_groups; i_gout++) {
      tab.fmu[gin][i_gout].resize(n_mu);
      for (int imu = 0; imu < n_mu; imu++) {
        tab.fmu[gin][i_gout][imu] =
          evaluate_legendre(leg.dist[gin][i_gout].size() - 1,
            leg.dist[gin][i_gout].data(), tab.mu[imu]);
      }

      // Ensure positivity
      for (auto& val : tab.fmu[gin][i_gout]) {
        if (val < 0.)
          val = 0.;
      }

      // Now re-normalize for numerical integration issues and to take care of
      // the above negative fix-up.  Also accrue the CDF
      double norm = 0.;
      tab.dist[gin][i_gout][0] = 0.;
      for (int imu = 1; imu < n_mu; imu++) {
        norm += 0.5 * tab.dmu *
                (tab.fmu[gin][i_gout][imu - 1] + tab.fmu[gin][i_gout][imu]);
        // incorporate to the CDF
        tab.dist[gin][i_gout][imu] = norm;
      }

      // now do the normalization
      if (norm > 0.) {
        for (int imu = 0; imu < n_mu; imu++) {
          tab.fmu[gin][i_gout][imu] /= norm;
          tab.dist[gin][i_gout][imu] /= norm;
        }
      }
    }
  }
}

} // namespace openmc
