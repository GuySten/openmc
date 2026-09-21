#include "openmc/physics_common.h"

#include "openmc/bep.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"

namespace openmc {

//==============================================================================
// RUSSIAN_ROULETTE
//==============================================================================

void russian_roulette(Particle& p, double weight_survive)
{
  if (weight_survive * prn(p.current_seed()) < p.wgt()) {
    p.wgt() = weight_survive;
  } else {
    p.wgt() = 0.;
  }
}

void apply_russian_roulette(Particle& p)
{
  // Exit if survival biasing is turned off
  if (!settings::survival_biasing)
    return;

  // A BEP shadow particle always takes the normalized branch, whatever the
  // user set. The absolute branch compares a weight against weight_cutoff
  // directly, which presumes a population born at about unit weight -- true
  // of the driver, false of a perturbation's tree, whose population can sit
  // decades below the branch site it grew from (measured at 2.3e-4 of the
  // reference for a photoneutron worth). Judged absolutely, such a tree is
  // entirely below a cutoff of 0.25 and every particle in it would be
  // rouletted up to weight_survive: unbiased, and ruinous -- 1 survivor in
  // 6900, each inflated by the same factor.
  //
  // Normalized, the same 0.25 and 1.0 mean what they say, because
  // run_one_tree() sets wgt_born to the weight a typical particle in the tree
  // is born at and a photoneutron gets its own birth weight. So survival
  // biasing needs no perturbation-specific settings: turning it on is enough.
  //
  // Driver particles are BEP_TRUNK and so never take this branch on account
  // of it, leaving a stock run bit-identical.
  if (settings::survival_normalization || p.bep_tree() != BEP_TRUNK) {
    if (p.wgt() < settings::weight_cutoff * p.wgt_born()) {
      russian_roulette(p, settings::weight_survive * p.wgt_born());
    }
  } else if (p.wgt() < settings::weight_cutoff) {
    russian_roulette(p, settings::weight_survive);
  }
}
} // namespace openmc
