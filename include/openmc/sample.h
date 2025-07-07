#ifndef OPENMC_SAMPLE_H
#define OPENMC_SAMPLE_H

#include "openmc/constants.h"
#include "openmc/math_functions.h"

namespace openmc {

struct Sample {
  
  uint64_t n;
  double s1;
  double s2;
  Sample() {n=0;s1=0.0;s2=0.0;}
  Sample(const vector<double>& v);
  
  // Unary operators
  Sample& operator+=(Sample);
  Sample& operator+=(double);
  Sample& operator-=(double);
  Sample& operator*=(double);
  Sample& operator/=(double);
  Sample& operator-();
  
  // Methods
  inline double mean() const { return (n>0) ? s1/n : 0.0 ; }
  inline double variance() const { return (n>1) ? (s2-s1*s1/n)/(n-1) : 0.0 ;}
  inline double std_dev() const { return (n>1) ? std::sqrt((s2-s1*s1/n)/(n-1)) : 0.0 ; }
  inline double confidence(double level) const { return (n>1) ? std_dev() * t_percentile((1.0 + level) / 2.0, n - 1) : 0.0;}
  std::pair<double, double> mean_stdev() const { return {mean(), std_dev()}; }
  inline double rel_err() const { return (s1 != 0.) ? std_dev() / std::abs(mean()) : INFTY; }
  void clear() { n=0; s1=0.0; s2=0.0; }
  void push_back(const double v);
};

// Binary operators
inline Sample operator+(Sample a, Sample b)
{
  return a += b;
}
inline Sample operator+(Sample a, double b)
{
  return a += b;
}
inline Sample operator+(double a, Sample b)
{
  return b += a;
}
inline Sample operator-(Sample a, double b)
{
  return a -= b;
}
inline Sample operator-(double a, Sample b)
{
  return b -= a;
}
inline Sample operator*(Sample a, double b)
{
  return a *= b;
}
inline Sample operator*(double a, Sample b)
{
  return b *= a;
}

inline Sample operator/(Sample a, double b)
{
  return a /= b;
}

}


#endif // OPENMC_SAMPLE_H
