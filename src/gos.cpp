#include "openmc/gos.h"

#include <algorithm> // for max, min
#include <cmath>

#include "openmc/constants.h"

namespace openmc {

namespace {

//! 2 pi r_e^2 m_e c^2 in [b eV], written from the constants already tabulated
constexpr double BOHR_RADIUS_CM =
  PLANCK_C * FINE_STRUCTURE / (2.0 * PI * MASS_ELECTRON_EV) * 1.0e-8;
constexpr double R_E = BOHR_RADIUS_CM / (FINE_STRUCTURE * FINE_STRUCTURE);
constexpr double COLLISION_CONST =
  2.0 * PI * 1.0e24 * R_E * R_E * MASS_ELECTRON_EV;

//! Moments of the close (binary) cross section over [w_lo, w_hi]
//
// Moller's cross section written for a projectile of total energy `ee`, whose
// antiderivatives are elementary. For a bound shell `ee` is E + U: the
// collision is with an electron that has to be paid for before it can be
// ejected.
void close_moments(double ee, double amol, double w_lo, double w_hi, double& m0,
  double& m1, double& m2)
{
  if (!(w_hi > w_lo) || !(w_lo > 0.0) || !(w_hi < ee))
    return;
  double ee_sq = ee * ee;
  m0 +=
    1.0 / (ee - w_hi) - 1.0 / (ee - w_lo) - 1.0 / w_hi + 1.0 / w_lo +
    (1.0 - amol) * std::log((ee - w_hi) * w_lo / ((ee - w_lo) * w_hi)) / ee +
    amol * (w_hi - w_lo) / ee_sq;
  m1 += std::log(w_hi / w_lo) + ee / (ee - w_hi) - ee / (ee - w_lo) +
        (2.0 - amol) * std::log((ee - w_hi) / (ee - w_lo)) +
        amol * (w_hi * w_hi - w_lo * w_lo) / (2.0 * ee_sq);
  m2 += (2.0 - amol) * (w_hi - w_lo) + w_hi * (2.0 * ee - w_hi) / (ee - w_hi) -
        w_lo * (2.0 * ee - w_lo) / (ee - w_lo) +
        (3.0 - amol) * ee * std::log((ee - w_hi) / (ee - w_lo)) +
        amol * (w_hi * w_hi * w_hi - w_lo * w_lo * w_lo) / (3.0 * ee_sq);
}

} // namespace

GosMoments gos_oscillator(
  double E, double u_b, double w_r, double delta, double w_cc)
{
  GosMoments m;
  constexpr double two_m = 2.0 * MASS_ELECTRON_EV;

  // A bound shell cannot take less than its ionisation energy; the conduction
  // oscillator, which has no binding, cannot take less than its resonance
  bool bound = u_b > 1.0e-3;
  double w_thr = bound ? u_b : w_r;
  if (E < w_thr + 1.0e-6)
    return m;

  double gamma = 1.0 + E / MASS_ELECTRON_EV;
  double gamma_sq = gamma * gamma;
  double beta_sq = (gamma_sq - 1.0) / gamma_sq;
  double k = COLLISION_CONST / beta_sq;
  double cp = std::sqrt(E * (E + two_m));
  double amol = std::pow(E / (E + MASS_ELECTRON_EV), 2);

  // Near threshold the resonance and the cutoff recoil are varied so the
  // cross section leaves it smoothly rather than stepping. Salvat's trick.
  double w_m, w_kp, q_kp, w_cmax, w_dmax, ee;
  if (bound) {
    w_m = 3.0 * w_r - 2.0 * u_b;
    if (E > w_m) {
      w_kp = w_r;
      q_kp = u_b;
    } else {
      w_kp = (E + 2.0 * u_b) / 3.0;
      q_kp = u_b * (E / w_m);
      w_m = E;
    }
    ee = E + u_b;
    w_cmax = 0.5 * ee;
    w_dmax = std::min(w_cmax, w_m);
  } else {
    w_m = E;
    w_kp = w_r;
    q_kp = w_r;
    ee = E;
    w_cmax = 0.5 * E;
    w_dmax = w_kp + 1.0;
  }

  // ==========================================================================
  // Distant interactions: the projectile passes at a distance and the atom
  // takes W_k with little momentum. The longitudinal part runs over the recoil
  // between its kinematic minimum and the resonance; the transverse part is
  // what the density effect screens away.
  double s_lon = 0.0;
  double s_tra = 0.0;
  if (w_dmax > w_thr + 1.0e-6) {
    double cpp = std::sqrt((E - w_kp) * (E - w_kp + two_m));
    double a = 4.0 * cp * cpp;
    double b = (cp - cpp) * (cp - cpp);
    double q_min;
    if (w_kp > 1.0e-6 * E) {
      q_min =
        std::sqrt(b + MASS_ELECTRON_EV * MASS_ELECTRON_EV) - MASS_ELECTRON_EV;
    } else {
      q_min = w_kp * w_kp / (beta_sq * two_m);
      q_min = q_min * (1.0 - q_min / two_m);
    }
    if (q_min < q_kp) {
      s_lon = std::log(q_kp * (q_min + two_m) / (q_min * (q_kp + two_m)));
      s_tra = std::max(0.0, std::log(gamma_sq) - beta_sq - delta);

      // Angular moments of the grouped distant collisions
      if (w_cc > w_thr && a > 0.0) {
        double ba = b / a;
        double mu1 = (q_kp * (q_kp + two_m) - b) / a;
        m.xs0_soft = std::log((mu1 + ba) / ba);
        m.xs1_soft = mu1 - ba * m.xs0_soft;
        m.xs2_soft = ba * ba * m.xs0_soft + 0.5 * mu1 * (mu1 - 2.0 * ba);
        m.xs0_soft /= w_kp;
        m.xs1_soft /= w_kp;
        m.xs2_soft /= w_kp;
        m.xs0_soft += s_tra / w_kp;
      }
    }
  }

  double s_dis = s_lon + s_tra;
  if (s_dis > 0.0) {
    if (bound) {
      // Inner-shell excitations, spread over a triangle from U to W_dmax
      // rather than left at a single energy
      double f0 = 1.0 / ((w_m - u_b) * (w_m - u_b));
      double f1 = 2.0 * f0 * s_dis / w_kp;
      auto triangle = [&](double lo, double hi, double& n0, double& n1,
                        double& n2) {
        n0 += f1 * (w_m * (hi - lo) - (hi * hi - lo * lo) / 2.0);
        n1 += f1 * (w_m * (hi * hi - lo * lo) / 2.0 -
                     (hi * hi * hi - lo * lo * lo) / 3.0);
        n2 += f1 * (w_m * (hi * hi * hi - lo * lo * lo) / 3.0 -
                     (hi * hi * hi * hi - lo * lo * lo * lo) / 4.0);
      };
      if (w_cc < u_b) {
        triangle(u_b, w_dmax, m.xs_hard, m.s_hard, m.w2_hard);
      } else {
        double lo, hi;
        if (w_cc > w_dmax) {
          lo = u_b;
          hi = w_dmax;
          triangle(lo, hi, m.xs_soft, m.s_soft, m.w2_soft);
        } else {
          triangle(w_cc, w_dmax, m.xs_hard, m.s_hard, m.w2_hard);
          lo = u_b;
          hi = w_cc;
          triangle(lo, hi, m.xs_soft, m.s_soft, m.w2_soft);
        }
        double f2 = f0 * (2.0 * w_m * (hi - lo) - (hi * hi - lo * lo));
        m.xs0_soft *= f2;
        m.xs1_soft *= f2;
        m.xs2_soft *= f2;
      }
    } else {
      // Outer shells: one delta oscillator at the resonance
      if (w_cc < w_kp) {
        m.s_hard += s_dis;
        m.xs_hard += s_dis / w_kp;
        m.w2_hard += s_dis * w_kp;
      } else {
        m.s_soft += s_dis;
        m.xs_soft += s_dis / w_kp;
        m.w2_soft += s_dis * w_kp;
      }
    }
  }

  // ==========================================================================
  // Close collisions: the binary cross section above the threshold
  if (w_cmax > w_thr + 1.0e-6) {
    if (w_cc < w_thr) {
      close_moments(ee, amol, w_thr, w_cmax, m.xs_hard, m.s_hard, m.w2_hard);
    } else if (w_cc > w_cmax) {
      close_moments(ee, amol, w_thr, w_cmax, m.xs_soft, m.s_soft, m.w2_soft);
    } else {
      close_moments(ee, amol, w_cc, w_cmax, m.xs_hard, m.s_hard, m.w2_hard);
      close_moments(ee, amol, w_thr, w_cc, m.xs_soft, m.s_soft, m.w2_soft);
    }
  }

  m.xs_soft *= k;
  m.s_soft *= k;
  m.w2_soft *= k;
  m.xs_hard *= k;
  m.s_hard *= k;
  m.w2_hard *= k;
  m.xs0_soft *= k;
  m.xs1_soft *= k;
  m.xs2_soft *= k;
  return m;
}

} // namespace openmc
