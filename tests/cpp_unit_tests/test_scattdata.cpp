#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openmc/math_functions.h"
#include "openmc/random_dist.h"
#include "openmc/scattdata.h"
#include "openmc/vector.h"

using namespace openmc;

namespace {

// Truncated Legendre expansion of the Henyey-Greenstein phase function,
// a_l = g^l
vector<double> henyey_greenstein(double g, int order)
{
  vector<double> a(order + 1);
  for (int l = 0; l <= order; ++l) {
    a[l] = std::pow(g, l);
  }
  return a;
}

} // namespace

TEST_CASE("Test beta_variate")
{
  uint64_t seed = 1;
  auto [a, b] = GENERATE(table<int, int>({{1, 1}, {1, 4}, {3, 1}, {2, 5}}));
  int n = 200000;
  double sum = 0.;
  double sum_sq = 0.;
  for (int i = 0; i < n; ++i) {
    double x = beta_variate(a, b, &seed);
    REQUIRE(x >= 0.);
    REQUIRE(x <= 1.);
    sum += x;
    sum_sq += x * x;
  }
  double mean = static_cast<double>(a) / (a + b);
  double var = a * b / ((a + b) * (a + b) * (a + b + 1.));
  REQUIRE_THAT(
    sum / n, Catch::Matchers::WithinAbs(mean, 5. * std::sqrt(var / n)));
  REQUIRE_THAT(
    sum_sq / n - (sum / n) * (sum / n), Catch::Matchers::WithinRel(var, 0.02));
}

TEST_CASE("Test LegendreBetaMixture reproduces positive expansions")
{
  vector<vector<double>> cases {{1.}, {1., 1. / 3.},
    {1.000e+0, 1.536e-1, 1.772e-2, 5.945e-4, 3.497e-5, 1.881e-5},
    {1., 0.6, 0.3, 0.1}, {1., 0., 0.4}, henyey_greenstein(0.8, 32)};

  for (const auto& coeffs : cases) {
    LegendreBetaMixture mixture(coeffs);
    int n = coeffs.size() - 1;

    // The mixture reproduces the expansion
    double fmax = 0.;
    for (int i = 0; i <= 2000; ++i) {
      double mu = -1. + i / 1000.;
      fmax = std::max(fmax, evaluate_legendre(n, coeffs.data(), mu));
    }
    for (int i = 0; i <= 2000; ++i) {
      double mu = -1. + i / 1000.;
      double f = evaluate_legendre(n, coeffs.data(), mu);
      REQUIRE_THAT(mixture.evaluate(mu),
        Catch::Matchers::WithinAbs(std::max(f, 0.), 1.e-9 * fmax));
    }

    // Sample means of the Legendre polynomials equal the coefficients
    uint64_t seed = 1;
    int n_samples = 100000;
    int n_moments = std::min(n, 3);
    vector<double> moments(n_moments + 1, 0.);
    vector<double> pn(n_moments + 1);
    for (int i = 0; i < n_samples; ++i) {
      double mu = mixture.sample(&seed);
      REQUIRE(std::abs(mu) <= 1.);
      calc_pn_c(n_moments, mu, pn.data());
      for (int l = 0; l <= n_moments; ++l) {
        moments[l] += pn[l] / n_samples;
      }
    }
    for (int l = 1; l <= n_moments; ++l) {
      // |P_l| <= 1, so 5 standard deviations is at most 5/sqrt(N)
      REQUIRE_THAT(moments[l],
        Catch::Matchers::WithinAbs(coeffs[l], 5. / std::sqrt(n_samples)));
    }
  }
}

TEST_CASE("Test LegendreBetaMixture sets negative parts to zero")
{
  // Negative near mu = -1, and an oscillating truncated expansion
  auto coeffs =
    GENERATE(vector<double> {1., 0.9, 0.5}, henyey_greenstein(0.9, 15));
  LegendreBetaMixture mixture(coeffs);
  int n = coeffs.size() - 1;

  // The mixture is the normalized positive part of the expansion
  int n_mu = 20001;
  double dmu = 2. / (n_mu - 1);
  vector<double> positive(n_mu);
  double norm = 0.;
  for (int i = 0; i < n_mu; ++i) {
    double mu = -1. + i * dmu;
    positive[i] = std::max(evaluate_legendre(n, coeffs.data(), mu), 0.);
    norm += (i == 0 || i == n_mu - 1 ? 0.5 : 1.) * dmu * positive[i];
  }
  for (int i = 0; i < n_mu; i += 100) {
    double mu = -1. + i * dmu;
    REQUIRE_THAT(mixture.evaluate(mu),
      Catch::Matchers::WithinAbs(positive[i] / norm, 1.e-4));
  }

  // Samples never fall where the expansion is negative
  uint64_t seed = 1;
  for (int i = 0; i < 10000; ++i) {
    double mu = mixture.sample(&seed);
    REQUIRE(evaluate_legendre(n, coeffs.data(), mu) > -1.e-6);
  }
}

TEST_CASE("Test LegendreBetaMixture without a positive part")
{
  // There is nothing to sample, so the distribution falls back to isotropic
  LegendreBetaMixture zero({0., 0.});
  REQUIRE_THAT(zero.evaluate(0.3), Catch::Matchers::WithinAbs(0.5, 1.e-14));
  uint64_t seed = 1;
  REQUIRE(std::abs(zero.sample(&seed)) <= 1.);
}

TEST_CASE("Test LegendreBetaMixture sample histogram")
{
  // Compare the fraction of samples in fine bins of mu with the probability
  // of each bin, which checks how the histogram part reuses its random number
  auto coeffs = GENERATE(vector<double> {1., 0.2},
    vector<double> {1., 0.9, 0.5}, henyey_greenstein(0.8, 32));
  LegendreBetaMixture mixture(coeffs);
  int n = coeffs.size() - 1;

  int n_bins = 200;
  int n_samples = 2000000;
  vector<double> counts(n_bins, 0.);
  uint64_t seed = 1;
  for (int i = 0; i < n_samples; ++i) {
    double mu = mixture.sample(&seed);
    int bin = std::min(static_cast<int>((mu + 1.) / 2. * n_bins), n_bins - 1);
    counts[bin] += 1.;
  }

  // Bin probabilities from the positive part of the expansion with Simpson's
  // rule
  vector<double> prob(n_bins);
  double total = 0.;
  for (int i = 0; i < n_bins; ++i) {
    double a = -1. + 2. * i / n_bins;
    double h = 2. / n_bins;
    double sum = 0.;
    int m = 20;
    for (int j = 0; j <= m; ++j) {
      double mu = a + h * j / m;
      double w = (j == 0 || j == m) ? 1. : (j % 2 == 1 ? 4. : 2.);
      sum += w * std::max(evaluate_legendre(n, coeffs.data(), mu), 0.);
    }
    prob[i] = sum * h / (3. * m);
    total += prob[i];
  }
  for (int i = 0; i < n_bins; ++i) {
    double expected = n_samples * prob[i] / total;
    REQUIRE_THAT(counts[i], Catch::Matchers::WithinAbs(expected,
                              5. * std::sqrt(expected) + 1.e-3 * expected));
  }
}

namespace {

// One incoming and one outgoing group with the given (weight, lower, upper)
// step triplets
ScattDataNEPSF one_group_steps(const vector<double>& triplets)
{
  ScattDataNEPSF data;
  tensor::Tensor<int> gmin({1}, 0);
  tensor::Tensor<int> gmax({1}, 0);
  double_2dvec mult {{1.}};
  double_3dvec coeffs {{triplets}};
  data.init(gmin, gmax, mult, coeffs);
  return data;
}

// Mean of P_l over [a, b], or P_l(a) if a == b
double mean_legendre(int l, double a, double b)
{
  vector<double> pa(l + 2);
  vector<double> pb(l + 2);
  calc_pn_c(l + 1, a, pa.data());
  calc_pn_c(l + 1, b, pb.data());
  if (b == a)
    return pa[l];
  if (l == 0)
    return 1.;
  return (pb[l + 1] - pb[l - 1] - pa[l + 1] + pa[l - 1]) /
         ((2 * l + 1) * (b - a));
}

} // namespace

TEST_CASE("Test ScattDataNEPSF moments and density")
{
  // Two steps and a discrete angle; weights sum to the cross section 2
  vector<double> triplets {0.5, -0.8, -0.2, 1.2, 0.3, 0.9, 0.3, 0.95, 0.95};
  auto data = one_group_steps(triplets);
  REQUIRE(data.get_order() == 5);
  REQUIRE_THAT(data.scattxs[0], Catch::Matchers::WithinRel(2., 1e-14));

  auto matrix = data.get_matrix(6);
  for (int l = 0; l <= 6; ++l) {
    double expected = 0.;
    for (int k = 0; k < 9; k += 3) {
      expected +=
        triplets[k] * mean_legendre(l, triplets[k + 1], triplets[k + 2]);
    }
    REQUIRE_THAT(matrix(0, 0, l), Catch::Matchers::WithinAbs(expected, 1e-13));
  }

  // The density of the steps, normalized over the transfer
  REQUIRE_THAT(
    data.calc_f(0, 0, -0.5), Catch::Matchers::WithinRel(0.25 / 0.6, 1e-14));
  REQUIRE_THAT(
    data.calc_f(0, 0, 0.5), Catch::Matchers::WithinRel(0.6 / 0.6, 1e-14));
  REQUIRE(data.calc_f(0, 0, 0.0) == 0.);
}

TEST_CASE("Test ScattDataNEPSF sampling")
{
  vector<double> triplets {0.5, -0.8, -0.2, 1.2, 0.3, 0.9, 0.3, 0.95, 0.95};
  auto data = one_group_steps(triplets);

  uint64_t seed = 1;
  int n = 400000;
  int n_point = 0;
  int n_gap = 0;
  vector<double> sum(4, 0.);
  vector<double> pn(4);
  for (int i = 0; i < n; ++i) {
    int gout;
    double mu;
    double wgt = 1.;
    data.sample(0, gout, mu, wgt, &seed);
    REQUIRE(gout == 0);
    REQUIRE(wgt == 1.);
    if (mu == 0.95)
      ++n_point;
    if ((mu > -0.2 && mu < 0.3) || mu < -0.8 || (mu > 0.9 && mu != 0.95))
      ++n_gap;
    calc_pn_c(3, mu, pn.data());
    for (int l = 0; l < 4; ++l)
      sum[l] += pn[l];
  }
  REQUIRE(n_gap == 0);
  REQUIRE_THAT(n_point / static_cast<double>(n),
    Catch::Matchers::WithinAbs(0.15, 5. * std::sqrt(0.15 * 0.85 / n)));
  auto matrix = data.get_matrix(3);
  for (int l = 1; l < 4; ++l) {
    REQUIRE_THAT(sum[l] / n,
      Catch::Matchers::WithinAbs(matrix(0, 0, l) / 2., 5. / std::sqrt(n)));
  }
}

TEST_CASE("Test ScattDataNEPSF combine concatenates steps")
{
  auto a = one_group_steps({1., -1., 0.});
  auto b = one_group_steps({2., 0.5, 0.7, 1., 0.9, 1.});
  ScattDataNEPSF combined;
  vector<ScattData*> those {&a, &b};
  combined.combine(those, {0.5, 2.});

  // 0.5 * 1 + 2 * 3
  REQUIRE_THAT(combined.scattxs[0], Catch::Matchers::WithinRel(6.5, 1e-14));
  REQUIRE(combined.get_order() == 3);

  // Moments are those of the weighted sum of the data
  auto ma = a.get_matrix(5);
  auto mb = b.get_matrix(5);
  auto mc = combined.get_matrix(5);
  for (int l = 0; l <= 5; ++l) {
    REQUIRE_THAT(mc(0, 0, l),
      Catch::Matchers::WithinAbs(0.5 * ma(0, 0, l) + 2. * mb(0, 0, l), 1e-13));
  }

  // Densities add as well, including where steps overlap
  REQUIRE_THAT(combined.calc_f(0, 0, 0.6),
    Catch::Matchers::WithinRel((2. * 2. / 0.2) / 6.5, 1e-13));
  REQUIRE_THAT(combined.calc_f(0, 0, -0.5),
    Catch::Matchers::WithinRel((0.5 * 1.) / 6.5, 1e-13));
}
