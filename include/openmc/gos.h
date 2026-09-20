//! \file gos.h
//! Sternheimer-Liljequist generalized oscillator strength model

#ifndef OPENMC_GOS_H
#define OPENMC_GOS_H

#include <cstdint>

#include "openmc/constants.h"

namespace openmc {

//! The free Moller and Bhabha differential cross sections, as functions of the
//! fraction eps = W/T of its kinetic energy the projectile transfers
//
//! Both carry the same leading constant, so only their shapes are written here
//! and the constant cancels wherever the two are divided. The coefficients are
//! PENELOPE's.
struct FreeCollision {
  double b1, b2, b3, b4; //!< Bhabha
  double amol, moller_c; //!< Moller

  explicit FreeCollision(double E)
  {
    double gamma = 1.0 + E / MASS_ELECTRON_EV;
    // ((gamma-1)/gamma)^2, the constant term of the Moller shape and the
    // common factor of every Bhabha coefficient
    amol = std::pow((gamma - 1.0) / gamma, 2);
    double g12 = (gamma + 1.0) * (gamma + 1.0);
    b1 = amol * (2.0 * g12 - 1.0) / (gamma * gamma - 1.0);
    b2 = amol * (3.0 + 1.0 / g12);
    b3 = amol * 2.0 * gamma * (gamma - 1.0) / g12;
    b4 = amol * (gamma - 1.0) * (gamma - 1.0) / g12;
    moller_c = (2.0 * gamma - 1.0) / (gamma * gamma);
  }

  double bhabha(double x) const
  {
    return (1.0 + x * (-b1 + x * (b2 + x * (-b3 + x * b4)))) / (x * x);
  }

  double moller(double x) const
  {
    double u = 1.0 - x;
    return 1.0 / (x * x) + 1.0 / (u * u) + amol - moller_c / (x * u);
  }

  //! Ratio of the two. It tends to 1 as x tends to 0, where both become
  //! Rutherford's 1/x^2, so it leaves the soft collisions -- which are almost
  //! all of them -- alone. Relativistically it is 1 - 2x to good accuracy.
  double ratio(double x) const { return this->bhabha(x) / this->moller(x); }
};

//! Most draws a rejection loop is allowed before it gives up
//
// Every loop here has a bound that holds, so this is not the mechanism that
// makes them terminate; it is there so that a cross section nobody has
// checked cannot hang a run.
constexpr int MAX_REJECTION {1000};

//==============================================================================
//! Moments of one oscillator's inelastic cross section, split at a cutoff
//!
//! Every quantity is per oscillator and per unit oscillator strength, in
//! barns and eV: multiply by \f$f_k\f$ and the atom density to get a
//! macroscopic rate. The soft and hard parts are the same differential cross
//! section either side of \f$W_{cc}\f$, which is what lets a condensed-history
//! step group one and sample the other without a seam between them.
//==============================================================================

struct GosMoments {
  double xs_soft {0.0};  //!< rate of the grouped collisions, [b]
  double s_soft {0.0};   //!< stopping power they carry, [b eV]
  double w2_soft {0.0};  //!< second moment of their loss, [b eV^2]
  double xs_hard {0.0};  //!< rate of the discrete collisions, [b]
  double s_hard {0.0};   //!< stopping power they carry, [b eV]
  double w2_hard {0.0};  //!< second moment of their loss, [b eV^2]
  double xs0_soft {0.0}; //!< grouped angular moments, orders 0 to 2, [b]
  double xs1_soft {0.0};
  double xs2_soft {0.0};
};

//! Integrated inelastic cross sections for one Sternheimer-Liljequist
//! oscillator, restricted to energy losses above and below a cutoff
//!
//! This is PENELOPE's EINaT1, and the model is Salvat's: each subshell is one
//! oscillator with an ionisation energy \f$U_k\f$ and a resonance energy
//! \f$W_k\f$, and a collision with it is either distant -- transferring
//! \f$W_k\f$ with little momentum, split into a longitudinal part and a
//! transverse part the density effect screens -- or close, which is the free
//! binary cross section restricted to transfers above the resonance.
//!
//! What makes it worth using rather than an evaluated spectrum is that the
//! resonance energies are fixed by \f$\sum_k f_k \ln W_k = \ln I\f$ and the
//! strengths by \f$\sum_k f_k = Z\f$. Those are the same two constraints that
//! determine the Bethe stopping power, so the model reproduces ICRU 37
//! analytically rather than by calibration: a single oscillator at \f$W = I\f$
//! gives the Bethe formula to five decimal places, which the unit tests check.
//!
//! The inner shells carry two refinements of Salvat's, both kept here. Their
//! resonance and cutoff recoil energies are varied near threshold so the cross
//! section rises smoothly out of it rather than stepping, and their distant
//! excitations are spread over a triangular distribution instead of sitting at
//! a single energy.
//!
//! \param[in] E Kinetic energy of the projectile in [eV]
//! \param[in] u_b Ionisation energy of the subshell in [eV]; zero or
//!   negligible marks the conduction oscillator, which has no binding
//! \param[in] w_r Resonance energy of the oscillator in [eV]
//! \param[in] delta Density-effect correction
//! \param[in] w_cc Soft cutoff in [eV]. Zero puts everything in the hard
//!   channel, which is single-event transport and is exact: every oscillator
//!   has a threshold, so the cross section stays finite there.
//! \param[in] positron Whether the projectile is a positron. Only the close
//!   collisions differ: the shape is Bhabha's rather than Moller's and the
//!   transfer runs to the whole kinetic energy, the two being
//!   distinguishable. The distant interactions are identical for the two
//!   charges, which is Salvat's observation and the reason this is one
//!   routine rather than two.
//! \return The moments above and below the cutoff
GosMoments gos_oscillator(double E, double u_b, double w_r, double delta,
  double w_cc, bool positron = false);

//==============================================================================
//! What one hard inelastic collision does
//==============================================================================

struct GosCollision {
  double w {0.0};        //!< energy the projectile loses, in [eV]
  double mu {1.0};       //!< cosine of its deflection
  double e_knock {0.0};  //!< kinetic energy of the ejected electron, in [eV]
  double mu_knock {1.0}; //!< cosine of the ejected electron's direction
  bool ionised {false};  //!< whether a vacancy was left in the subshell
};

//! Sample a hard collision with one oscillator
//!
//! PENELOPE's EINa. Three outcomes share the oscillator's hard cross section:
//! a close collision, which is the binary one and takes the whole transfer as
//! recoil; a distant longitudinal one, which hands over little momentum and
//! whose recoil is distributed as \f$1/Q(Q+2mc^2)\f$; and a distant
//! transverse one, which is the part the density effect screens and leaves
//! the projectile undeflected.
//!
//! The energy loss is the oscillator's resonance energy for an outer shell
//! and is drawn from the same triangle the moments were integrated over for
//! an inner one, so what is sampled and what was counted are the same
//! distribution.
//!
//! \param[in] E Kinetic energy of the projectile in [eV]
//! \param[in] u_b Ionisation energy of the subshell in [eV]
//! \param[in] w_r Resonance energy of the oscillator in [eV]
//! \param[in] delta Density-effect correction
//! \param[in] w_cc Soft cutoff in [eV]
//! \param[in] positron Whether the projectile is a positron
//! \param[inout] seed Pseudorandom number seed pointer
GosCollision sample_gos_collision(double E, double u_b, double w_r,
  double delta, double w_cc, bool positron, uint64_t* seed);

} // namespace openmc

#endif // OPENMC_GOS_H
