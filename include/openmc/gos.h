//! \file gos.h
//! Sternheimer-Liljequist generalized oscillator strength model

#ifndef OPENMC_GOS_H
#define OPENMC_GOS_H

namespace openmc {

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
//! \return The moments above and below the cutoff
GosMoments gos_oscillator(
  double E, double u_b, double w_r, double delta, double w_cc);

} // namespace openmc

#endif // OPENMC_GOS_H
