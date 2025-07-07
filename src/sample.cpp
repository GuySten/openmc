#include "openmc/sample.h"

namespace openmc {

//==============================================================================
// Sample implementation
//==============================================================================


Sample::Sample(const vector<double>& v)
{
  Sample sample;
  for (auto& x: v) {
    sample.push_back(x);
  }
}

Sample& Sample::operator+=(Sample v)
{
  n+=v.n;
  s1 += v.s1;
  s2 += v.s2;
  return *this;
}

Sample& Sample::operator+=(double v)
{
  s2 += 2*v*s1+v*v*n;
  s1 += v*n;
  return *this;
}

Sample& Sample::operator-=(double v)
{
  s2 -= 2*v*s1-v*v*n;
  s1 -= v*n;
  return *this;
}

Sample& Sample::operator*=(double v)
{
  s1 *= v;
  s2 *= v*v;
  return *this;
}

Sample& Sample::operator/=(double v)
{
  s1 /= v;
  s2 /= v*v;
  return *this;
}

Sample& Sample::operator-()
{
  s1 *= -1;
  return *this;
}

void Sample::push_back(const double v)
{
  ++n;
  s2+=v*v;
#pragma omp atomic
  s1+=v;  
}

} // namespace openmc
