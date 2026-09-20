#include "openmc/gos.h"

#include <algorithm> // for max, min
#include <cmath>

#include "openmc/constants.h"
#include "openmc/random_lcg.h"

namespace openmc {

namespace {

//! 2 pi r_e^2 m_e c^2 in [b eV], written from the constants already tabulated
constexpr double BOHR_RADIUS_CM =
  PLANCK_C * FINE_STRUCTURE / (2.0 * PI * MASS_ELECTRON_EV) * 1.0e-8;
constexpr double R_E = BOHR_RADIUS_CM / (FINE_STRUCTURE * FINE_STRUCTURE);
constexpr double COLLISION_CONST =
  2.0 * PI * 1.0e24 * R_E * R_E * MASS_ELECTRON_EV;

//! Moments of the close Bhabha cross section over [w_lo, w_hi]
//
// The positron is distinguishable from the electron it strikes, so the shape
// is Bhabha's and the transfer runs to the whole kinetic energy rather than
// half of it. The four coefficients are the same ones FreeCollision carries.
void close_moments_positron(double E, double amol, double w_lo, double w_hi,
  double& m0, double& m1, double& m2)
{
  if (!(w_hi > w_lo) || !(w_lo > 0.0))
    return;
  double gamma = 1.0 + E / MASS_ELECTRON_EV;
  double g12 = (gamma + 1.0) * (gamma + 1.0);
  double b1 = amol * (2.0 * g12 - 1.0) / (gamma * gamma - 1.0);
  double b2 = amol * (3.0 + 1.0 / g12);
  double b3 = amol * 2.0 * gamma * (gamma - 1.0) / g12;
  double b4 = amol * (gamma - 1.0) * (gamma - 1.0) / g12;
  double e2 = E * E;
  double e3 = e2 * E;
  double e4 = e3 * E;
  double d1 = w_hi - w_lo;
  double d2 = w_hi * w_hi - w_lo * w_lo;
  double d3 = w_hi * w_hi * w_hi - w_lo * w_lo * w_lo;
  double d4 = std::pow(w_hi, 4) - std::pow(w_lo, 4);
  double d5 = std::pow(w_hi, 5) - std::pow(w_lo, 5);
  double lg = std::log(w_hi / w_lo);
  m0 += 1.0 / w_lo - 1.0 / w_hi - b1 * lg / E + b2 * d1 / e2 -
        b3 * d2 / (2.0 * e3) + b4 * d3 / (3.0 * e4);
  m1 += lg - b1 * d1 / E + b2 * d2 / (2.0 * e2) - b3 * d3 / (3.0 * e3) +
        b4 * d4 / (4.0 * e4);
  m2 += d1 - b1 * d2 / (2.0 * E) + b2 * d3 / (3.0 * e2) - b3 * d4 / (4.0 * e3) +
        b4 * d5 / (5.0 * e4);
}

//! Moments of the close Moller cross section over [w_lo, w_hi]
//
// Written for a projectile of total energy `ee`, whose antiderivatives are
// elementary. For a bound shell `ee` is E + U: the collision is with an
// electron that has to be paid for before it can be ejected.
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
  double E, double u_b, double w_r, double delta, double w_cc, bool positron)
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
    ee = positron ? E : E + u_b;
    w_cmax = positron ? E : 0.5 * ee;
    w_dmax = std::min(w_cmax, w_m);
  } else {
    w_m = E;
    w_kp = w_r;
    q_kp = w_r;
    ee = E;
    w_cmax = positron ? E : 0.5 * E;
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

      // Angular moments of the grouped distant collisions.
      //
      // The recoil is written in x = (1 - mu)/2, in which the momentum
      // transfer is (cq)^2 = b + a x and the distribution is dx/(x + b/a).
      // The integrals T_n = \int x^n dx/(x + b/a) are elementary, and x_max
      // is where the cutoff recoil puts it.
      //
      // The step wants the transport cross sections, which are moments of
      // 1 - mu and (3/2)(1 - mu^2) rather than of x. Since 1 - mu = 2x and
      // 1 - mu^2 = 4x(1 - x),
      //
      //     sigma_1 = 2 T_1,      sigma_2 = 6 (T_1 - T_2),
      //
      // and those are what is stored. PENELOPE applies the same two factors
      // in the routine that calls this one; leaving them out halves the
      // inelastic share of sigma_1 and empties sigma_2 altogether, against an
      // elastic contribution that is already in the transport convention.
      if (w_cc > w_thr && a > 0.0) {
        double ba = b / a;
        double x_max = (q_kp * (q_kp + two_m) - b) / a;
        double t0 = std::log((x_max + ba) / ba);
        double t1 = x_max - ba * t0;
        double t2 = ba * ba * t0 + 0.5 * x_max * (x_max - 2.0 * ba);
        m.xs0_soft = (t0 + s_tra) / w_kp;
        m.xs1_soft = 2.0 * t1 / w_kp;
        m.xs2_soft = 6.0 * (t1 - t2) / w_kp;
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
    auto close = [&](double lo, double hi, double& n0, double& n1, double& n2) {
      if (positron) {
        close_moments_positron(E, amol, lo, hi, n0, n1, n2);
      } else {
        close_moments(ee, amol, lo, hi, n0, n1, n2);
      }
    };
    if (w_cc < w_thr) {
      close(w_thr, w_cmax, m.xs_hard, m.s_hard, m.w2_hard);
    } else if (w_cc > w_cmax) {
      close(w_thr, w_cmax, m.xs_soft, m.s_soft, m.w2_soft);
    } else {
      close(w_cc, w_cmax, m.xs_hard, m.s_hard, m.w2_hard);
      close(w_thr, w_cc, m.xs_soft, m.s_soft, m.w2_soft);
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

GosCollision sample_gos_collision(double E, double u_b, double w_r,
  double delta, double w_cc, bool positron, uint64_t* seed)
{
  GosCollision c;
  c.w = 0.0;
  constexpr double two_m = 2.0 * MASS_ELECTRON_EV;

  bool bound = u_b > 1.0e-3;
  double w_thr = std::max(w_cc, bound ? u_b : w_r);
  if (E < w_thr + 1.0e-6)
    return c;

  // The same threshold trick the moments were integrated under, so that what
  // is sampled here is the distribution they counted
  bool distant = true;
  double w_m, w_kp, q_kp, ee, w_cmax, w_dmax;
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
    if (w_cc > w_m)
      distant = false;
    ee = positron ? E : E + u_b;
    w_cmax = positron ? E : 0.5 * ee;
    w_dmax = std::min(w_m, positron ? E : 0.5 * (E + u_b));
    if (w_thr > w_dmax)
      distant = false;
  } else {
    if (w_cc > w_r)
      distant = false;
    w_kp = w_r;
    q_kp = w_r;
    w_m = E;
    ee = E;
    w_cmax = positron ? E : 0.5 * E;
    w_dmax = w_kp + 1.0;
  }

  double rb = E + two_m;
  double gamma = 1.0 + E / MASS_ELECTRON_EV;
  double gamma_sq = gamma * gamma;
  double beta_sq = (gamma_sq - 1.0) / gamma_sq;
  double amol = std::pow((gamma - 1.0) / gamma, 2);
  double cps = E * rb;
  double cp = std::sqrt(cps);

  // Partial cross sections of this oscillator, in the same units as each
  // other; only their ratios are used
  double x_lon = 0.0;
  double x_tra = 0.0;
  double cpp = 0.0;
  double cpps = 0.0;
  double q_min = 0.0;
  if (distant) {
    cpps = (E - w_kp) * (E - w_kp + two_m);
    cpp = std::sqrt(cpps);
    if (w_kp > 1.0e-6 * E) {
      q_min = std::sqrt(
                (cp - cpp) * (cp - cpp) + MASS_ELECTRON_EV * MASS_ELECTRON_EV) -
              MASS_ELECTRON_EV;
    } else {
      q_min = w_kp * w_kp / (beta_sq * two_m);
      q_min = q_min * (1.0 - q_min / two_m);
    }
    if (q_min < q_kp) {
      x_lon =
        std::log(q_kp * (q_min + two_m) / (q_min * (q_kp + two_m))) / w_kp;
      x_tra = std::max(0.0, std::log(gamma_sq) - beta_sq - delta) / w_kp;
      if (bound) {
        double f0 = (w_dmax - w_thr) * (w_m + w_m - w_dmax - w_thr) /
                    ((w_m - u_b) * (w_m - u_b));
        x_lon *= f0;
        x_tra *= f0;
      }
    }
  }

  // Close collisions, from the same antiderivative the moments used
  double x_close = 0.0;
  if (w_cmax > w_thr) {
    double m0 = 0.0, m1 = 0.0, m2 = 0.0;
    if (positron) {
      close_moments_positron(E, amol, w_thr, w_cmax, m0, m1, m2);
    } else {
      close_moments(ee, amol, w_thr, w_cmax, m0, m1, m2);
    }
    x_close = std::max(0.0, m0);
  }

  double total = x_close + x_lon + x_tra;
  if (total < 1.0e-35)
    return c;

  double xi = prn(seed) * total;

  // ==========================================================================
  // Close collision: the transfer comes from the binary shape by rejection
  // against its 1/W^2 envelope, which is what the shape is built around
  if (xi < x_close) {
    // Sample the 1/W^2 the shape is built around, which inverts exactly, and
    // carry the rest by rejection. Times x^2 the shape is 1 at x = 0 and
    // monotone from there, so the bound is the end point.
    FreeCollision fc {E};
    double w = w_thr;
    double inv_lo = 1.0 / w_thr;
    double inv_hi = 1.0 / w_cmax;
    double x_hi = w_cmax / ee;
    double u_hi = 1.0 - x_hi;
    double bound_shape =
      positron ? 1.0 + fc.b1 * x_hi + fc.b2 * x_hi * x_hi +
                   fc.b3 * x_hi * x_hi * x_hi + fc.b4 * std::pow(x_hi, 4)
               : 1.0 + x_hi * x_hi / (u_hi * u_hi) + fc.amol * x_hi * x_hi;
    for (int it = 0; it < MAX_REJECTION; ++it) {
      w = 1.0 / (inv_lo - prn(seed) * (inv_lo - inv_hi));
      double x = w / ee;
      double shape = positron ? fc.bhabha(x) : fc.moller(x);
      if (prn(seed) * bound_shape <= shape * x * x)
        break;
    }
    c.w = w;
    c.mu = std::sqrt((E - w) * rb / (E * (rb - w)));
    c.e_knock = bound ? w - u_b : w;
    c.mu_knock = std::sqrt(w * rb / (E * (w + two_m)));
    c.ionised = bound;
    return c;
  }

  // ==========================================================================
  // Distant interaction. The loss is the resonance for an outer shell and is
  // drawn from the triangle for an inner one.
  double w = w_kp;
  if (bound) {
    w = w_m -
        std::sqrt((w_m - w_thr) * (w_m - w_thr) -
                  prn(seed) * (w_dmax - w_thr) * (w_m + w_m - w_dmax - w_thr));
  }
  c.w = w;
  c.e_knock = bound ? w - u_b : w;
  c.ionised = bound;

  if (xi < x_close + x_lon) {
    // Longitudinal: the recoil is distributed as 1/(Q(Q+2mc^2)) between its
    // kinematic minimum and the resonance, which inverts in closed form
    double qs = q_min / (1.0 + q_min / two_m);
    double q = qs / (std::pow((qs / q_kp) * (1.0 + q_kp / two_m), prn(seed)) -
                      qs / two_m);
    double q_tot = q * (q + two_m);
    c.mu = std::min(1.0, (cpps + cps - q_tot) / (2.0 * cp * cpp));
    c.mu_knock = std::min(
      1.0, 0.5 * (w_kp * (E + rb - w_kp) + q_tot) / std::sqrt(cps * q_tot));
    return c;
  }

  // Transverse: no momentum handed over, so nothing is deflected. This is the
  // branch the density effect screens away.
  c.mu = 1.0;
  c.mu_knock = 1.0;
  return c;
}

} // namespace openmc
