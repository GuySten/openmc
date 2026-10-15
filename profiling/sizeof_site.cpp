// Print sizeof(SourceSite) and sizeof(ParticleData) for a given set of
// profiling build flags. Built by profiling/sizeof_site.sh.
#include "openmc/particle_data.h"
#include <cstdio>
int main()
{
  std::printf("sizeof(SourceSite)=%zu sizeof(ParticleData)=%zu "
              "sizeof(NuclideMicroXS)=%zu sizeof(ElementMicroXS)=%zu\n",
    sizeof(openmc::SourceSite), sizeof(openmc::ParticleData),
    sizeof(openmc::NuclideMicroXS), sizeof(openmc::ElementMicroXS));
}
