#ifndef OPENMC_VOLUME_OCTREE_MATH_H
#define OPENMC_VOLUME_OCTREE_MATH_H

//! \file volume_octree_math.h
//! \brief Geometry and logic kernels for the deterministic volume calculation.
//!
//! Deliberately free of every OpenMC dependency except Position, so that the
//! part of this feature that can be numerically wrong -- the range bounds and
//! the three-valued evaluator -- is unit-testable without building OpenMC.
//! See tests/cpp_unit_tests/test_volume_octree_math.cpp.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "openmc/position.h"
#include "openmc/vector.h"

namespace openmc {

//==============================================================================
//! Octree cell: an oriented box.
//==============================================================================
//
// The box is always expressed in the frame of the universe being traversed,
// because that is the frame its surfaces are defined in.  Descending through a
// rotated fill leaves the box a box -- rigid motions preserve boxes and
// volumes -- but no longer axis-aligned, so the frame is carried explicitly:
//
//     p(u) = center + sum_k u_k * axis[k],   u_k in [-half[k], half[k]]
//
// `rotated` is a fast path.  While it is false, axis[] is the identity and the
// transforms below collapse to the axis-aligned case.

struct OctBox {
  Position center;
  Position axis[3] {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
  double half[3] {0., 0., 0.};
  bool rotated {false};

  static OctBox from_aabb(Position lo, Position hi)
  {
    OctBox b;
    b.center = 0.5 * (lo + hi);
    b.half[0] = 0.5 * (hi.x - lo.x);
    b.half[1] = 0.5 * (hi.y - lo.y);
    b.half[2] = 0.5 * (hi.z - lo.z);
    return b;
  }

  double volume() const { return 8.0 * half[0] * half[1] * half[2]; }

  Position corner(int i) const
  {
    return center + ((i & 1) ? half[0] : -half[0]) * axis[0] +
           ((i & 2) ? half[1] : -half[1]) * axis[1] +
           ((i & 4) ? half[2] : -half[2]) * axis[2];
  }

  //! Subdivide along the axes flagged in `mask` (bit k selects axis k).
  //!
  //! `i` runs over [0, 2^popcount(mask)). The default splits all three, giving
  //! the usual eight octants. Splitting a subset matters more than it sounds:
  //! see axis_irrelevant() below.
  OctBox child(int i, int mask = 0b111) const
  {
    OctBox c = *this;
    int bit = 0;
    for (int k = 0; k < 3; ++k) {
      if (!((mask >> k) & 1))
        continue;
      double h = 0.5 * half[k];
      c.half[k] = h;
      c.center = c.center + (((i >> bit) & 1) ? h : -h) * axis[k];
      ++bit;
    }
    return c;
  }

  static int n_children(int mask)
  {
    return 1 << (((mask >> 0) & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1));
  }

  //! The same physical box re-expressed in a child universe's frame.
  //!
  //! OpenMC applies r_child = R * (r_parent - t) when descending a fill
  //! (geometry.cpp: `coord.r() -= c.translation_` then `coord.rotate(...)`),
  //! so the centre follows the same rule and the frame axes take the rotation
  //! without the translation.  Pass an empty rotation for a translation-only
  //! fill.
  template<typename T>
  OctBox transformed(const Position& translation, const T& rotation) const
  {
    OctBox b = *this;
    b.center = center - translation;
    if (!rotation.empty()) {
      b.center = b.center.rotate(rotation);
      for (int k = 0; k < 3; ++k)
        b.axis[k] = axis[k].rotate(rotation);
      b.rotated = true;
    }
    return b;
  }
};

//==============================================================================
//! Surface as a quadratic form: q(p) = p^T A p + b . p + c
//==============================================================================
//
// A is symmetric, stored as (xx, yy, zz, xy, xz, yz).

struct QuadricForm {
  double A[6] {0., 0., 0., 0., 0., 0.};
  Position b {0., 0., 0.};
  double c {0.};
};

enum class BoxSense : uint8_t { UNKNOWN = 0, NEGATIVE, POSITIVE, BOTH };

inline Position sym_mul(const double A[6], const Position& v)
{
  return {A[0] * v.x + A[3] * v.y + A[4] * v.z,
    A[3] * v.x + A[1] * v.y + A[5] * v.z, A[4] * v.x + A[5] * v.y + A[2] * v.z};
}

inline double quad_eval(const QuadricForm& q, const Position& p)
{
  return p.dot(sym_mul(q.A, p)) + q.b.dot(p) + q.c;
}

inline BoxSense from_range(double lo, double hi)
{
  if (hi < 0.0)
    return BoxSense::NEGATIVE;
  if (lo > 0.0)
    return BoxSense::POSITIVE;
  return BoxSense::BOTH;
}

//! Exact range of a*t^2 + b*t over t in [-h, h].
inline void quad1d_range(double a, double b, double h, double& lo, double& hi)
{
  double v1 = a * h * h - b * h;
  double v2 = a * h * h + b * h;
  lo = std::min(v1, v2);
  hi = std::max(v1, v2);
  if (a != 0.0) {
    double t = -b / (2.0 * a);
    if (std::abs(t) <= h) {
      double v3 = -b * b / (4.0 * a);
      lo = std::min(lo, v3);
      hi = std::max(hi, v3);
    }
  }
}

//! Exact range of t^2 for t in [a, b].
inline void sq_range(double a, double b, double& lo, double& hi)
{
  if (a >= 0.0) {
    lo = a * a;
    hi = b * b;
  } else if (b <= 0.0) {
    lo = b * b;
    hi = a * a;
  } else {
    lo = 0.0;
    hi = std::max(a * a, b * b);
  }
}

//! Exact range of the distance from (c1, c2) to the rectangle
//! [l1, h1] x [l2, h2].
inline void radial_range(double c1, double c2, double l1, double h1, double l2,
  double h2, double& rmin, double& rmax)
{
  auto near = [](double c, double l, double h) {
    if (c < l)
      return l - c;
    if (c > h)
      return c - h;
    return 0.0;
  };
  auto far = [](double c, double l, double h) {
    return std::max(std::abs(c - l), std::abs(c - h));
  };
  double n1 = near(c1, l1, h1), n2 = near(c2, l2, h2);
  double f1 = far(c1, l1, h1), f2 = far(c2, l2, h2);
  rmin = std::sqrt(n1 * n1 + n2 * n2);
  rmax = std::sqrt(f1 * f1 + f2 * f2);
}

//! Range of a quadratic form over a box.
//!
//! Substituting p = o + sum_k u_k a_k gives another quadratic, in u, over the
//! *centred* box [-h, h]^3:
//!
//!     A'_kl = a_k . A a_l     b'_k = a_k . (2 A o + b)     c' = q(o)
//!
//! The box is a product set, so a separable quadratic -- no cross terms -- has
//! an exactly computable range: sum the exact 1-D ranges.  That covers planes,
//! axis-aligned cylinders, spheres and cones against unrotated boxes, and
//! spheres against any box.  Cross terms get the centred bound
//! |2 A'_kl u_k u_l| <= 2 |A'_kl| h_k h_l, which is sound and O(h^2), so
//! refinement kills it quickly.  Centring is what makes that usable; the same
//! extension on an off-origin box is far looser.
inline void form_range(
  const QuadricForm& q, const OctBox& box, double& lo, double& hi)
{
  double App[3], bp[3], off[3]; // off = |A'_01|, |A'_02|, |A'_12|

  Position g = 2.0 * sym_mul(q.A, box.center) + q.b;

  if (!box.rotated) {
    App[0] = q.A[0];
    App[1] = q.A[1];
    App[2] = q.A[2];
    bp[0] = g.x;
    bp[1] = g.y;
    bp[2] = g.z;
    off[0] = std::abs(q.A[3]);
    off[1] = std::abs(q.A[4]);
    off[2] = std::abs(q.A[5]);
  } else {
    Position Aa[3];
    for (int k = 0; k < 3; ++k)
      Aa[k] = sym_mul(q.A, box.axis[k]);
    for (int k = 0; k < 3; ++k) {
      App[k] = box.axis[k].dot(Aa[k]);
      bp[k] = box.axis[k].dot(g);
    }
    off[0] = std::abs(box.axis[0].dot(Aa[1]));
    off[1] = std::abs(box.axis[0].dot(Aa[2]));
    off[2] = std::abs(box.axis[1].dot(Aa[2]));
  }

  double cp = quad_eval(q, box.center);
  lo = cp;
  hi = cp;
  for (int k = 0; k < 3; ++k) {
    double l, u;
    quad1d_range(App[k], bp[k], box.half[k], l, u);
    lo += l;
    hi += u;
  }

  double m = 2.0 * (off[0] * box.half[0] * box.half[1] +
                     off[1] * box.half[0] * box.half[2] +
                     off[2] * box.half[1] * box.half[2]);
  lo -= m;
  hi += m;
}

//! Does the surface's classification over this box vary along box axis k?
//!
//! Substituting p = o + sum_l u_l a_l, the coefficient of every term involving
//! u_k vanishes exactly when A a_k = 0 (which kills A'_kl for all l, by
//! symmetry) and a_k . (2 A o + b) = 0. Then splitting the box along a_k
//! produces two children with identical classifications for this surface.
//!
//! This is worth checking because reactor geometry is overwhelmingly extruded:
//! every ZCylinder in a pin lattice is independent of z, so an octree splits
//! the axial direction purely to no purpose. Measured on a pincell, skipping
//! those splits changes the node count per level from x4.03 to x2.13, which
//! turns the cost-versus-tolerance scaling from second order into first.
inline bool axis_irrelevant(
  const QuadricForm& q, const OctBox& box, int k, const Position& g)
{
  Position Aa = sym_mul(q.A, box.axis[k]);
  if (Aa.x != 0.0 || Aa.y != 0.0 || Aa.z != 0.0)
    return false;
  return box.axis[k].dot(g) == 0.0;
}

//! The gradient term reused by axis_irrelevant(): g = 2 A o + b.
inline Position form_gradient(const QuadricForm& q, const OctBox& box)
{
  return 2.0 * sym_mul(q.A, box.center) + q.b;
}

//! Exact range of a torus equation over an unrotated box.
//!
//! The equation splits into an axial term and a term in the radial distance
//! from the axis; both ranges are exact and the terms are independent, so the
//! sum is exact.  `ax` selects the torus axis (0 = x, 1 = y, 2 = z).  A, B, C
//! follow OpenMC's convention: axial^2/B^2 + (rho - A)^2/C^2 - 1.
inline void torus_range(int ax, Position tc, double A, double B, double C,
  const OctBox& box, double& lo, double& hi)
{
  Position blo {box.center.x - box.half[0], box.center.y - box.half[1],
    box.center.z - box.half[2]};
  Position bhi {box.center.x + box.half[0], box.center.y + box.half[1],
    box.center.z + box.half[2]};

  const double lo_a[3] {blo.x, blo.y, blo.z};
  const double hi_a[3] {bhi.x, bhi.y, bhi.z};
  const double tc_a[3] {tc.x, tc.y, tc.z};

  int r1 = (ax + 1) % 3, r2 = (ax + 2) % 3;

  double a_lo, a_hi;
  sq_range(lo_a[ax] - tc_a[ax], hi_a[ax] - tc_a[ax], a_lo, a_hi);
  a_lo /= B * B;
  a_hi /= B * B;

  double rmin, rmax;
  radial_range(
    tc_a[r1], tc_a[r2], lo_a[r1], hi_a[r1], lo_a[r2], hi_a[r2], rmin, rmax);
  double t_lo, t_hi;
  sq_range(rmin - A, rmax - A, t_lo, t_hi);
  t_lo /= C * C;
  t_hi /= C * C;

  lo = a_lo + t_lo - 1.0;
  hi = a_hi + t_hi - 1.0;
}

//==============================================================================
//! Base-case integrators
//==============================================================================
//
// The Box integrator alone gives slack ~ A*h against cost ~ A/h^2, i.e. the
// same eps^-2 rate as Monte Carlo. Refinement will never fix that. The way out
// is to stop subdividing once a box's straddling surfaces are simple enough to
// integrate in closed form, which moves the residual slack off the 2-D boundary
// surface entirely. These are the closed forms.

//! Fraction of a box's volume charged to slack whenever a closed-form
//! integrator is used instead of refinement.
//!
//! The integrators are exact in exact arithmetic, but the halfspace subset sum
//! loses ~1e-11 relative to cancellation (measured by box-subdivision
//! additivity). Charging a guard keeps "lower <= V <= lower + slack" literally
//! true rather than true-up-to-rounding, at a cost far below any tolerance
//! anyone will ask for.
constexpr double INTEGRATOR_GUARD {1.0e-10};

//! Recognise q as a plane: q < 0 is the halfspace n . p <= d.
inline bool as_plane(const QuadricForm& q, Position& n, double& d)
{
  for (int k = 0; k < 6; ++k)
    if (q.A[k] != 0.0)
      return false;
  if (q.b.x == 0.0 && q.b.y == 0.0 && q.b.z == 0.0)
    return false;
  n = q.b;
  d = -q.c;
  return true;
}

//! Recognise q as an axis-aligned plane. `below` is true when the negative
//! halfspace is the low side of `value` along `axis`.
inline bool as_axis_plane(
  const QuadricForm& q, int& axis, double& value, bool& below)
{
  Position n;
  double d;
  if (!as_plane(q, n, d))
    return false;
  const double nb[3] {n.x, n.y, n.z};
  int nz = -1;
  for (int k = 0; k < 3; ++k) {
    if (nb[k] == 0.0)
      continue;
    if (nz >= 0)
      return false;
    nz = k;
  }
  axis = nz;
  value = d / nb[nz];
  below = nb[nz] > 0.0;
  return true;
}

//! Recognise q as an axis-aligned cylinder. `axis` is the cylinder axis;
//! (c1, c2) is the centre in the two radial coordinates, in cyclic order.
inline bool as_axis_cylinder(
  const QuadricForm& q, int& axis, double& c1, double& c2, double& r)
{
  if (q.A[3] != 0.0 || q.A[4] != 0.0 || q.A[5] != 0.0)
    return false;
  int zero = -1;
  for (int k = 0; k < 3; ++k) {
    if (q.A[k] == 0.0) {
      if (zero >= 0)
        return false;
      zero = k;
    } else if (q.A[k] != 1.0) {
      return false;
    }
  }
  if (zero < 0)
    return false;
  const double bb[3] {q.b.x, q.b.y, q.b.z};
  if (bb[zero] != 0.0)
    return false;
  axis = zero;
  int r1 = (zero + 1) % 3, r2 = (zero + 2) % 3;
  c1 = -0.5 * bb[r1];
  c2 = -0.5 * bb[r2];
  double r2sq = c1 * c1 + c2 * c2 - q.c;
  if (r2sq <= 0.0)
    return false;
  r = std::sqrt(r2sq);
  return true;
}

//! Area of the disk of radius r centred at the origin intersected with
//! {u <= x, v <= y}. Finite because the disk is.
//!
//! Built from cap(t) = area{u > t} and corner(x,y) = area{u > x, v > y} by
//! inclusion-exclusion, which makes the rectangle case below a clean four-term
//! combination with no case analysis of its own.
inline double disk_quadrant_area(double x, double y, double r)
{
  const double A = M_PI * r * r;

  auto cap = [r, A](double t) {
    if (t <= -r)
      return A;
    if (t >= r)
      return 0.0;
    return r * r * std::acos(t / r) - t * std::sqrt(r * r - t * t);
  };

  auto corner = [&](double cx, double cy) {
    if (cx >= r || cy >= r)
      return 0.0;
    if (cx <= -r && cy <= -r)
      return A;
    if (cx <= -r)
      return cap(cy);
    if (cy <= -r)
      return cap(cx);
    if (cx * cx + cy * cy <= r * r) {
      // The corner point is inside the circle: a triangle plus the circular
      // segment beyond the chord joining the two arc crossings.
      double sy = std::sqrt(r * r - cx * cx);
      double sx = std::sqrt(r * r - cy * cy);
      double tri = 0.5 * (sy - cy) * (sx - cx);
      double a1 = std::atan2(sy, cx);
      double a2 = std::atan2(cy, sx);
      double alpha = a1 - a2;
      return tri + 0.5 * r * r * (alpha - std::sin(alpha));
    }
    // Corner outside the circle.
    if (cx >= 0.0 && cy >= 0.0)
      return 0.0; // nearest point of the quadrant is already outside
    if (cx < 0.0 && cy >= 0.0)
      return cap(cy); // the vertical cut misses the horizontal cap
    if (cy < 0.0 && cx >= 0.0)
      return cap(cx);
    // Both negative and outside: the two excluded caps are disjoint.
    return A - cap(-cx) - cap(-cy);
  };

  return A - cap(x) - cap(y) + corner(x, y);
}

//! Exact area of a disk intersected with an axis-aligned rectangle.
inline double circle_rect_area(
  double cx, double cy, double r, double x0, double x1, double y0, double y1)
{
  if (r <= 0.0)
    return 0.0;
  auto g = [&](double x, double y) {
    return disk_quadrant_area(x - cx, y - cy, r);
  };
  double a = g(x1, y1) - g(x0, y1) - g(x1, y0) + g(x0, y0);
  return a > 0.0 ? a : 0.0;
}

//==============================================================================
//! 2-D convex polygon helpers, for cross-sections that are not rectangles.
//==============================================================================

//! Signed area of a polygon given as parallel x/y arrays.
inline double poly_area(const vector<double>& px, const vector<double>& py)
{
  double a2 = 0.0;
  int n = static_cast<int>(px.size());
  for (int k = 0; k < n; ++k) {
    int k2 = (k + 1) % n;
    a2 += px[k] * py[k2] - px[k2] * py[k];
  }
  return 0.5 * a2;
}

//! Clip a convex polygon to the halfplane a*x + b*y <= c (Sutherland-Hodgman).
inline void clip_poly(
  vector<double>& px, vector<double>& py, double a, double b, double c)
{
  vector<double> qx, qy;
  int n = static_cast<int>(px.size());
  qx.reserve(n + 2);
  qy.reserve(n + 2);
  for (int k = 0; k < n; ++k) {
    int k2 = (k + 1) % n;
    double f1 = a * px[k] + b * py[k] - c;
    double f2 = a * px[k2] + b * py[k2] - c;
    if (f1 <= 0.0) {
      qx.push_back(px[k]);
      qy.push_back(py[k]);
    }
    if ((f1 < 0.0 && f2 > 0.0) || (f1 > 0.0 && f2 < 0.0)) {
      double t = f1 / (f1 - f2);
      qx.push_back(px[k] + t * (px[k2] - px[k]));
      qy.push_back(py[k] + t * (py[k2] - py[k]));
    }
  }
  px.swap(qx);
  py.swap(qy);
}

//! Exact area of a disk intersected with a polygon.
//!
//! Sums, over each polygon edge, the signed area of disk ∩ triangle(O, a, b):
//! the edge is split where it crosses the circle, and each resulting chord is
//! either a triangle (chord inside the disk) or a circular sector (outside).
//! Generalises circle_rect_area to cross-sections cut by oblique planes -- a
//! hex assembly wall around a fuel pin, for instance -- which the rectangle
//! version cannot express.
inline double circle_poly_area(double cx, double cy, double r,
  const vector<double>& px, const vector<double>& py)
{
  if (r <= 0.0 || px.size() < 3)
    return 0.0;
  double r2 = r * r;
  int n = static_cast<int>(px.size());
  double total = 0.0;

  for (int k = 0; k < n; ++k) {
    int k2 = (k + 1) % n;
    double ax = px[k] - cx, ay = py[k] - cy;
    double bx = px[k2] - cx, by = py[k2] - cy;

    // Split the edge where it crosses the circle.
    double ex = bx - ax, ey = by - ay;
    double qa = ex * ex + ey * ey;
    double ts[4] {0.0, 1.0, 1.0, 1.0};
    int nt = 2;
    if (qa > 0.0) {
      double qb = 2.0 * (ax * ex + ay * ey);
      double qc = ax * ax + ay * ay - r2;
      double disc = qb * qb - 4.0 * qa * qc;
      if (disc > 0.0) {
        double sq = std::sqrt(disc);
        double t1 = (-qb - sq) / (2.0 * qa), t2 = (-qb + sq) / (2.0 * qa);
        if (t1 > 0.0 && t1 < 1.0)
          ts[nt++] = t1;
        if (t2 > 0.0 && t2 < 1.0 && t2 > t1)
          ts[nt++] = t2;
      }
    }
    std::sort(ts, ts + nt);

    for (int i = 0; i + 1 < nt; ++i) {
      double u0x = ax + ts[i] * ex, u0y = ay + ts[i] * ey;
      double u1x = ax + ts[i + 1] * ex, u1y = ay + ts[i + 1] * ey;
      double mx = 0.5 * (u0x + u1x), my = 0.5 * (u0y + u1y);
      double cross = u0x * u1y - u1x * u0y;
      // Bias the inside/outside decision toward "outside" (sector).
      //
      // A circle exactly TANGENT to a polygon edge is the knife-edge case: the
      // edge is outside the disk everywhere but one point, so the correct
      // contribution is the sector, yet the chord midpoint sits at distance
      // exactly r and a non-strict test picks the triangle. That is not exotic
      // -- a fuel pin inscribed in a hex cell is tangent to all six walls, and
      // the error was 5% of the disk area. Biasing costs nothing on genuine
      // secants: if the midpoint is inside by a relative margin delta, triangle
      // and sector differ by O(r^2 delta^1.5), which at delta = 1e-12 is below
      // rounding.
      if (mx * mx + my * my <= r2 * (1.0 - 1.0e-12)) {
        total += 0.5 * cross; // chord lies inside: triangle
      } else {
        double dot = u0x * u1x + u0y * u1y;
        total += 0.5 * r2 * std::atan2(cross, dot); // sector
      }
    }
  }
  return std::abs(total);
}

//! Exact volume of a box intersected with the halfspace {n . p <= d}.
//!
//! Works for an oriented box: project the normal onto the box frame, shift the
//! box to [0, L]^3, flip signs so every coefficient is non-negative, then apply
//! the standard slab formula
//!
//!   V = 1/(K! prod c_k) sum_{S} (-1)^{|S|} max(0, D - sum_{k in S} c_k L_k)^K
//!
//! over the K axes the plane actually varies along; axes it does not vary along
//! factor straight out. Returns false when a coefficient is small but not zero,
//! where the 1/prod(c_k) is ill-conditioned -- the caller then subdivides,
//! which is slower but keeps the bracket honest.
inline bool box_halfspace_volume(
  const OctBox& box, const Position& n, double d, double& vol)
{
  double c[3], L[3];
  double D = d - n.dot(box.center);
  for (int k = 0; k < 3; ++k) {
    double m = n.dot(box.axis[k]);
    L[k] = 2.0 * box.half[k];
    c[k] = std::abs(m);
    D += c[k] * box.half[k]; // shift to w in [0, L], flipping sign if m < 0
  }

  double max_cl = 0.0;
  for (int k = 0; k < 3; ++k)
    max_cl = std::max(max_cl, c[k] * L[k]);

  int active[3], K = 0;
  double outer = 1.0;
  for (int k = 0; k < 3; ++k) {
    if (c[k] == 0.0) {
      outer *= L[k];
    } else if (c[k] * L[k] < 1.0e-10 * max_cl) {
      return false; // ill-conditioned; let the caller refine instead
    } else {
      active[K++] = k;
    }
  }

  if (K == 0) {
    vol = (D >= 0.0) ? outer : 0.0;
    return true;
  }

  double denom = 1.0;
  for (int i = 0; i < K; ++i)
    denom *= c[active[i]];
  for (int i = 2; i <= K; ++i)
    denom *= i; // K!

  double sum = 0.0;
  for (int mask = 0; mask < (1 << K); ++mask) {
    double t = D;
    int bits = 0;
    for (int i = 0; i < K; ++i) {
      if ((mask >> i) & 1) {
        t -= c[active[i]] * L[active[i]];
        ++bits;
      }
    }
    if (t <= 0.0)
      continue;
    double term = 1.0;
    for (int e = 0; e < K; ++e)
      term *= t;
    sum += (bits % 2 ? -term : term);
  }

  vol = outer * sum / denom;
  if (vol < 0.0)
    vol = 0.0;
  double vmax = outer * L[active[0]] * (K > 1 ? L[active[1]] : 1.0) *
                (K > 2 ? L[active[2]] : 1.0);
  if (vol > vmax)
    vol = vmax;
  return true;
}

//! A surface of revolution about an axis-aligned axis. At axial coordinate w
//! the negative side is the annulus r_in(w) < rho < r_out(w).
//!
//! Quadrics -- spheres, axis-aligned cones and cylinders -- give r_in = 0 and
//! r_out^2 = -alpha (w - w0)^2 - K. Tori give a genuine annulus,
//! r_out = A + C s, r_in = max(0, A - C s) with s = sqrt(1 - ((w-w0)/B)^2).
//!
//! Both radii are monotone on each side of w0, and in OPPOSITE directions for
//! the torus, so the cross-sectional area -- which grows with r_out and shrinks
//! with r_in -- is monotone on each side either way. That is exactly the
//! property axis_sym_slice_bounds needs, so tori come along for free.
struct AxisSymProfile {
  enum class Kind : uint8_t { QUADRIC, TORUS };
  Kind kind {Kind::QUADRIC};
  int axis;
  double c1, c2; //!< centre in the two radial coordinates, cyclic order
  double w0;
  double alpha {0.}, K {0.};     //!< QUADRIC
  double A {0.}, B {0.}, C {0.}; //!< TORUS

  //! In-plane rotation taking the radial coordinates onto the ellipse's own
  //! principal axes. Zero for a diagonal quadric.
  //!
  //! A cross term COUPLING THE TWO RADIAL AXES (an ellipse tilted in its own
  //! plane) is removable by a rotation about the symmetry axis, which leaves
  //! the axis and any cross-section polygon intact -- so it costs nothing.
  //! Cross terms coupling the symmetry axis to a radial one are a different
  //! matter: removing those tilts the axis away from the box, and the
  //! cross-section then varies along it. Those are still refused.
  double ct {1.0}, st {0.0};

  //! Radial scalings. The cross-section of a diagonal quadric is an ELLIPSE,
  //! not a circle -- but scaling u = x*sx, v = y*sy with sx = sqrt(A_xx),
  //! sy = sqrt(A_yy) turns the ellipse into a circle while leaving an
  //! axis-aligned rectangle a rectangle, so the whole disk-against-polygon
  //! machinery applies unchanged and areas divide by sx*sy. That is what
  //! extends this from spheres, cones and circular cylinders to ellipsoids,
  //! elliptic cylinders, elliptic cones and any diagonal quadric.
  double sx {1.0}, sy {1.0};

  //! Map a point in the two radial coordinates into the frame where the
  //! cross-section is a circle: rotate onto the principal axes, then scale.
  //! Both are affine, so a polygon stays a polygon and areas divide by sx*sy.
  void to_circle_frame(double x, double y, double& u, double& v) const
  {
    u = (x * ct + y * st) * sx;
    v = (-x * st + y * ct) * sy;
  }

  void radii(double w, double& r_in, double& r_out) const
  {
    double t = w - w0;
    if (kind == Kind::QUADRIC) {
      double r2 = -alpha * t * t - K;
      r_in = 0.0;
      r_out = r2 > 0.0 ? std::sqrt(r2) : 0.0;
      return;
    }
    double u = 1.0 - (t / B) * (t / B);
    if (u <= 0.0) {
      r_in = r_out = 0.0;
      return;
    }
    double sc = C * std::sqrt(u);
    r_out = A + sc;
    r_in = A - sc;
    if (r_in < 0.0)
      r_in = 0.0;
  }

  double radius(double w) const
  {
    double a, b;
    radii(w, a, b);
    return b;
  }
};

//! Recognise any DIAGONAL quadric whose cross-section perpendicular to some
//! axis is an ellipse: spheres, circular and elliptic cylinders, circular and
//! elliptic cones, ellipsoids, elliptic paraboloids.
//!
//! The axis is chosen so the other two diagonal entries are positive. A zero
//! entry is preferred when present, since that is the cylinder case and puts
//! the profile in its simplest form.
//! Off-diagonal entry A_ij of the packed symmetric matrix (xx,yy,zz,xy,xz,yz).
inline double quad_off(const QuadricForm& q, int i, int j)
{
  if (i == j)
    return q.A[i];
  int a = std::min(i, j), b = std::max(i, j);
  if (a == 0 && b == 1)
    return q.A[3];
  if (a == 0 && b == 2)
    return q.A[4];
  return q.A[5];
}

//! Recognise any quadric whose cross-section perpendicular to some axis is an
//! ellipse: spheres, ellipsoids, circular and elliptic cylinders and cones,
//! including ones TILTED WITHIN THE CROSS-SECTION PLANE.
//!
//! The axis must be an eigenvector, i.e. uncoupled from both radial
//! coordinates. The remaining 2x2 radial block is then diagonalised by a
//! rotation about that axis, which leaves the axis and any cross-section
//! polygon intact.
inline bool as_axis_symmetric(const QuadricForm& q, AxisSymProfile& p)
{
  int axis = -1;
  for (int k = 0; k < 3; ++k) {
    int r1 = (k + 1) % 3, r2 = (k + 2) % 3;
    // The axis has to be uncoupled from both radial directions.
    if (quad_off(q, k, r1) != 0.0 || quad_off(q, k, r2) != 0.0)
      continue;
    // Diagonalise the radial block and require both eigenvalues positive, so
    // the cross-section is a bounded ellipse.
    double a = q.A[r1], b = q.A[r2], c = quad_off(q, r1, r2);
    double mid = 0.5 * (a + b);
    double rad = std::sqrt(0.25 * (a - b) * (a - b) + c * c);
    if (mid - rad <= 0.0)
      continue;
    // Prefer an axis with no quadratic term of its own: that is a cylinder,
    // the simplest form.
    if (axis < 0 || q.A[k] == 0.0)
      axis = k;
    if (q.A[k] == 0.0)
      break;
  }
  if (axis < 0)
    return false;

  int r1 = (axis + 1) % 3, r2 = (axis + 2) % 3;
  double a = q.A[r1], b = q.A[r2], c = quad_off(q, r1, r2);
  double mid = 0.5 * (a + b);
  double rad = std::sqrt(0.25 * (a - b) * (a - b) + c * c);
  double l1 = mid + rad, l2 = mid - rad;
  double theta = (c == 0.0 && a >= b) ? 0.0 : 0.5 * std::atan2(2.0 * c, a - b);

  const double bb[3] {q.b.x, q.b.y, q.b.z};
  double alpha = q.A[axis];

  p.kind = AxisSymProfile::Kind::QUADRIC;
  p.axis = axis;
  p.ct = std::cos(theta);
  p.st = std::sin(theta);
  p.sx = std::sqrt(l1);
  p.sy = std::sqrt(l2);

  // Linear term in the principal frame, then complete the squares there.
  double g1 = bb[r1] * p.ct + bb[r2] * p.st;
  double g2 = -bb[r1] * p.st + bb[r2] * p.ct;
  p.c1 = -0.5 * g1 / l1;
  p.c2 = -0.5 * g2 / l2;

  if (alpha == 0.0) {
    if (bb[axis] != 0.0)
      return false;
    p.w0 = 0.0;
  } else {
    p.w0 = -0.5 * bb[axis] / alpha;
  }
  p.alpha = alpha;
  p.K = q.c - l1 * p.c1 * p.c1 - l2 * p.c2 * p.c2 - alpha * p.w0 * p.w0;
  return true;
}

//! Torus profile. OpenMC's convention is
//! axial^2/B^2 + (rho - A)^2/C^2 - 1, with the torus axis along `axis`.
inline AxisSymProfile torus_profile(
  int axis, const Position& centre, double A, double B, double C)
{
  AxisSymProfile p;
  p.kind = AxisSymProfile::Kind::TORUS;
  p.axis = axis;
  const double cc[3] {centre.x, centre.y, centre.z};
  p.c1 = cc[(axis + 1) % 3];
  p.c2 = cc[(axis + 2) % 3];
  p.w0 = cc[axis];
  p.A = A;
  p.B = B;
  p.C = C;
  return p;
}

//! Rigorous bounds on the volume of box ∩ {q < 0} for a surface of revolution.
//!
//! Slicing perpendicular to the symmetry axis makes every cross-section a disk
//! against a rectangle, which circle_rect_area handles exactly. The axial
//! integral is not elementary, but R(w) is monotone on each side of w0 and the
//! area is monotone in R, so on a monotone slice the true area lies between the
//! two endpoint areas: the left and right Riemann sums BRACKET the integral.
//! That is what keeps this an integrator rather than an approximation -- the
//! spread is charged to slack, so the guarantee survives.
//!
//! Spread falls as 1/N at cost N, i.e. first order, against subdivision's
//! second-order eps^-2. That is the whole point for spheres and cones, which
//! no closed form here covers.
//! As axis_sym_slice_bounds, but with an explicit cross-section polygon in the
//! two radial coordinates. The polygon must not vary along the axis, which is
//! why only planes PARALLEL to the symmetry axis may be folded into it.
inline void axis_sym_slice_bounds_poly(const AxisSymProfile& p, double w_lo,
  double w_hi, const vector<double>& px, const vector<double>& py, int n_slices,
  double& lo, double& hi)
{
  double breaks[3] {w_lo, std::min(std::max(p.w0, w_lo), w_hi), w_hi};
  lo = 0.0;
  hi = 0.0;

  // Work in the scaled radial coordinates, where the ellipse is a circle, then
  // divide the area back. An axis-aligned scaling maps a polygon to a polygon,
  // so the cross-section survives unchanged in kind.
  vector<double> sxv(px.size()), syv(py.size());
  for (size_t k = 0; k < px.size(); ++k)
    p.to_circle_frame(px[k], py[k], sxv[k], syv[k]);
  double jac = p.sx * p.sy;
  double cu = p.c1 * p.sx, cv = p.c2 * p.sy; // centre is already principal

  // Fast path. circle_rect_area costs a few atan2 calls; circle_poly_area does
  // a quadratic solve and up to three sub-chords per edge. The cross-section is
  // an axis-aligned rectangle in the overwhelming majority of boxes, and this
  // runs once per slice with slice counts in the millions -- worth 4 lines.
  bool is_rect = sxv.size() == 4 && sxv[0] == sxv[3] && sxv[1] == sxv[2] &&
                 syv[0] == syv[1] && syv[2] == syv[3];
  double rx0 = 0, rx1 = 0, ry0 = 0, ry1 = 0;
  if (is_rect) {
    rx0 = std::min(sxv[0], sxv[1]);
    rx1 = std::max(sxv[0], sxv[1]);
    ry0 = std::min(syv[0], syv[2]);
    ry1 = std::max(syv[0], syv[2]);
  }

  auto disk = [&](double r) {
    return is_rect ? circle_rect_area(cu, cv, r, rx0, rx1, ry0, ry1)
                   : circle_poly_area(cu, cv, r, sxv, syv);
  };

  auto area = [&](double w) {
    double r_in, r_out;
    p.radii(w, r_in, r_out);
    double a = disk(r_out);
    if (r_in > 0.0)
      a -= disk(r_in);
    return a > 0.0 ? a / jac : 0.0;
  };

  for (int seg = 0; seg < 2; ++seg) {
    double s0 = breaks[seg], s1 = breaks[seg + 1];
    if (s1 <= s0)
      continue;
    double dw = (s1 - s0) / n_slices;
    double a_prev = area(s0);
    for (int i = 1; i <= n_slices; ++i) {
      double a_next = area(s0 + i * dw);
      lo += std::min(a_prev, a_next) * dw;
      hi += std::max(a_prev, a_next) * dw;
      a_prev = a_next;
    }
  }
}

inline void axis_sym_slice_bounds(const OctBox& box, const AxisSymProfile& p,
  int n_slices, double& lo, double& hi)
{
  double blo[3] {box.center.x - box.half[0], box.center.y - box.half[1],
    box.center.z - box.half[2]};
  double bhi[3] {box.center.x + box.half[0], box.center.y + box.half[1],
    box.center.z + box.half[2]};

  int a = p.axis, r1 = (a + 1) % 3, r2 = (a + 2) % 3;
  double w_lo = blo[a], w_hi = bhi[a];
  vector<double> px {blo[r1], bhi[r1], bhi[r1], blo[r1]};
  vector<double> py {blo[r2], blo[r2], bhi[r2], bhi[r2]};
  axis_sym_slice_bounds_poly(p, w_lo, w_hi, px, py, n_slices, lo, hi);
}

//! Volume of a bounded intersection of halfspaces n_i . p <= d_i.
//!
//! Each face is the plane's own 2-D convex polygon, obtained by starting from a
//! large square in that plane and clipping it against every other halfspace;
//! the volume then follows from the divergence theorem,
//! V = (1/3) sum_f (n_f . (p_f - p0)) * area_f, for any reference p0.
//!
//! Chosen over vertex enumeration because it degrades gracefully: a redundant
//! or coincident plane simply yields a zero-area face instead of a degenerate
//! triple that has to be detected and skipped. Requires the set to be bounded,
//! which the six box halfspaces guarantee.
inline double convex_halfspace_volume(
  const vector<Position>& nrm, const vector<double>& off, double scale)
{
  int m = static_cast<int>(nrm.size());
  double total = 0.0;
  vector<double> px, py, qx, qy;

  // Normalise, then drop exact duplicates.
  //
  // Two coincident halfspaces each contribute a full face to the divergence
  // sum, so the face is counted twice and the volume comes out too large. This
  // is not exotic: a volume calculation's bounding box routinely has faces
  // lying exactly on the model's own bounding planes, which produced a 4/3
  // error on a hex prism. Near-coincident planes are fine without this -- each
  // clips the other's face to zero area -- so only exact coincidence needs
  // handling, but the comparison carries a tolerance anyway.
  vector<Position> nh(m);
  vector<double> dh(m);
  vector<char> live(m, 1);
  for (int i = 0; i < m; ++i) {
    double len = std::sqrt(nrm[i].dot(nrm[i]));
    if (len == 0.0) {
      live[i] = 0;
      continue;
    }
    nh[i] = nrm[i] / len;
    dh[i] = off[i] / len;
  }
  for (int i = 0; i < m; ++i) {
    if (!live[i])
      continue;
    for (int j = 0; j < i; ++j) {
      if (!live[j])
        continue;
      Position dn = nh[i] - nh[j];
      if (dn.dot(dn) < 1.0e-24 &&
          std::abs(dh[i] - dh[j]) < 1.0e-12 * (1.0 + std::abs(dh[i]))) {
        // Keep the tighter of the two; they are equal to tolerance anyway.
        if (dh[i] < dh[j])
          live[j] = 0;
        else
          live[i] = 0;
        break;
      }
    }
  }

  for (int i = 0; i < m; ++i) {
    if (!live[i])
      continue;
    Position ni = nh[i];
    double di = dh[i];
    Position pi = di * ni;

    // Orthonormal basis in the plane.
    Position t =
      (std::abs(ni.x) < 0.9) ? Position {1., 0., 0.} : Position {0., 1., 0.};
    Position u = t - ni.dot(t) * ni;
    u = u / std::sqrt(u.dot(u));
    Position v = ni.cross(u);

    px = {-scale, scale, scale, -scale};
    py = {-scale, -scale, scale, scale};

    for (int j = 0; j < m && !px.empty(); ++j) {
      if (j == i || !live[j])
        continue;
      double a = nh[j].dot(u), b = nh[j].dot(v);
      double c = dh[j] - nh[j].dot(pi); // keep a*x + b*y <= c
      qx.clear();
      qy.clear();
      int n = static_cast<int>(px.size());
      for (int k = 0; k < n; ++k) {
        int k2 = (k + 1) % n;
        double f1 = a * px[k] + b * py[k] - c;
        double f2 = a * px[k2] + b * py[k2] - c;
        if (f1 <= 0.0) {
          qx.push_back(px[k]);
          qy.push_back(py[k]);
        }
        if ((f1 < 0.0 && f2 > 0.0) || (f1 > 0.0 && f2 < 0.0)) {
          double s = f1 / (f1 - f2);
          qx.push_back(px[k] + s * (px[k2] - px[k]));
          qy.push_back(py[k] + s * (py[k2] - py[k]));
        }
      }
      px.swap(qx);
      py.swap(qy);
    }

    int n = static_cast<int>(px.size());
    if (n < 3)
      continue;
    double area2 = 0.0, cx = 0.0, cy = 0.0;
    for (int k = 0; k < n; ++k) {
      int k2 = (k + 1) % n;
      double cr = px[k] * py[k2] - px[k2] * py[k];
      area2 += cr;
      cx += (px[k] + px[k2]) * cr;
      cy += (py[k] + py[k2]) * cr;
    }
    double area = 0.5 * std::abs(area2);
    if (area <= 0.0)
      continue;
    cx /= 3.0 * area2;
    cy /= 3.0 * area2;
    Position centroid = pi + cx * u + cy * v;
    total += ni.dot(centroid) * area;
  }
  return total / 3.0;
}

//! Volume of box ∩ {n_i . p <= d_i} for any number of planes and any box
//! orientation. Generalises box_halfspace_volume, which stays as the fast path
//! for the single-plane case.
inline double box_planes_volume(
  const OctBox& box, const vector<Position>& nrm, const vector<double>& off)
{
  vector<Position> n;
  vector<double> d;
  n.reserve(nrm.size() + 6);
  d.reserve(off.size() + 6);
  for (int k = 0; k < 3; ++k) {
    n.push_back(box.axis[k]);
    d.push_back(box.axis[k].dot(box.center) + box.half[k]);
    n.push_back(-1.0 * box.axis[k]);
    d.push_back(-(box.axis[k].dot(box.center) - box.half[k]));
  }
  for (size_t i = 0; i < nrm.size(); ++i) {
    n.push_back(nrm[i]);
    d.push_back(off[i]);
  }
  double diag =
    std::sqrt(box.half[0] * box.half[0] + box.half[1] * box.half[1] +
              box.half[2] * box.half[2]);
  double scale = 4.0 * (diag + std::sqrt(box.center.dot(box.center)) + 1.0);
  double v = convex_halfspace_volume(n, d, scale);
  if (v < 0.0)
    v = 0.0;
  if (v > box.volume())
    v = box.volume();
  return v;
}

//! Do two revolved profiles describe concentric surfaces, i.e. agree on
//! everything but the constant term? Compared exactly: surfaces built from the
//! same centre give bit-identical values, and a near miss should fall back to
//! refinement rather than be grouped wrongly.
inline bool same_revolved_geometry(
  const AxisSymProfile& a, const AxisSymProfile& b)
{
  return a.kind == AxisSymProfile::Kind::QUADRIC &&
         b.kind == AxisSymProfile::Kind::QUADRIC && a.axis == b.axis &&
         a.ct == b.ct && a.st == b.st && a.sx == b.sx && a.sy == b.sy &&
         a.c1 == b.c1 && a.c2 == b.c2 && a.w0 == b.w0 && a.alpha == b.alpha;
}

//! Rigorous bounds for an ANNULAR region between two concentric revolved
//! surfaces that differ only in their constant term.
//!
//! Nested surfaces are the case subdivision cannot rescue. Two curved surfaces
//! that cross transversally leave an ambiguous set that is a 1-D curve, so
//! refinement drives slack as L*h^2 at cost L/h -- measured, and good. Two
//! CONCENTRIC surfaces bounding a thin shell leave an ambiguous set that stays
//! 2-D until the box edge drops below the shell thickness, so refinement
//! crawls at A*h until then. A 0.008 cm gap on a 0.4 cm pin is six extra
//! levels, ~4000x the boxes.
//!
//! Members share axis, rotation, scaling, centre and w0, differing only in K,
//! so R_i(w)^2 = -alpha (w-w0)^2 - K_i and the radii are nested in a fixed
//! order independent of w. A sign combination therefore collapses to
//! R_b(w) < rho < R_a(w) with K_a > K_b, exactly as for coaxial cylinders.
//!
//! The two area terms are bounded INDEPENDENTLY per slice. Their difference is
//! not monotone in general -- both radii shrink together away from w0, so the
//! difference can move either way -- but each term separately is monotone, and
//! summing independent monotone bounds stays sound at the same O(1/N) rate.
inline void axis_sym_annulus_bounds(const AxisSymProfile& p, bool has_a,
  double K_a, bool has_b, double K_b, double w_lo, double w_hi,
  const vector<double>& px, const vector<double>& py, int n_slices, double& lo,
  double& hi)
{
  vector<double> sxv(px.size()), syv(py.size());
  for (size_t k = 0; k < px.size(); ++k)
    p.to_circle_frame(px[k], py[k], sxv[k], syv[k]);
  double jac = p.sx * p.sy;
  double cu = p.c1 * p.sx, cv = p.c2 * p.sy;
  double full = std::abs(poly_area(sxv, syv));

  bool is_rect = sxv.size() == 4 && sxv[0] == sxv[3] && sxv[1] == sxv[2] &&
                 syv[0] == syv[1] && syv[2] == syv[3];
  double rx0 = 0, rx1 = 0, ry0 = 0, ry1 = 0;
  if (is_rect) {
    rx0 = std::min(sxv[0], sxv[1]);
    rx1 = std::max(sxv[0], sxv[1]);
    ry0 = std::min(syv[0], syv[2]);
    ry1 = std::max(syv[0], syv[2]);
  }
  auto disk = [&](double r) {
    if (r <= 0.0)
      return 0.0;
    return is_rect ? circle_rect_area(cu, cv, r, rx0, rx1, ry0, ry1)
                   : circle_poly_area(cu, cv, r, sxv, syv);
  };
  auto radius = [&](double K, double w) {
    double t = w - p.w0;
    double r2 = -p.alpha * t * t - K;
    return r2 > 0.0 ? std::sqrt(r2) : 0.0;
  };
  auto term_a = [&](double w) { return has_a ? disk(radius(K_a, w)) : full; };
  auto term_b = [&](double w) { return has_b ? disk(radius(K_b, w)) : 0.0; };

  // Split at the extremum of the radii so each term is monotone per segment.
  double breaks[3] {w_lo, std::min(std::max(p.w0, w_lo), w_hi), w_hi};
  lo = 0.0;
  hi = 0.0;
  for (int seg = 0; seg < 2; ++seg) {
    double s0 = breaks[seg], s1 = breaks[seg + 1];
    if (s1 <= s0)
      continue;
    double dw = (s1 - s0) / n_slices;
    double a_prev = term_a(s0), b_prev = term_b(s0);
    for (int i = 1; i <= n_slices; ++i) {
      double w = s0 + i * dw;
      double a_next = term_a(w), b_next = term_b(w);
      double l = std::min(a_prev, a_next) - std::max(b_prev, b_next);
      double u = std::max(a_prev, a_next) - std::min(b_prev, b_next);
      if (l > 0.0)
        lo += l * dw;
      if (u > 0.0)
        hi += u * dw;
      a_prev = a_next;
      b_prev = b_next;
    }
  }
  lo /= jac;
  hi /= jac;
}

//! Cross-section of a box perpendicular to coordinate axis `axis`, as a polygon
//! in the two radial coordinates, plus the axial range.
//!
//! Only valid when one of the box's frame axes is parallel to `axis`: then the
//! section is the same polygon at every height. Otherwise a tilted box's
//! section changes as you move along the axis, which breaks the monotonicity
//! the slice bounds rest on, and this returns false. An unrotated box always
//! qualifies; so does the common case of a fill rotated ABOUT the axis its
//! cylinders run along.
inline bool box_axis_section(const OctBox& box, int axis, vector<double>& px,
  vector<double>& py, double& w_lo, double& w_hi)
{
  int r1 = (axis + 1) % 3, r2 = (axis + 2) % 3;
  Position e {0., 0., 0.};
  (axis == 0 ? e.x : (axis == 1 ? e.y : e.z)) = 1.0;

  int k_ax = -1;
  for (int k = 0; k < 3; ++k) {
    double d = std::abs(box.axis[k].dot(e));
    if (d > 1.0 - 1.0e-12) {
      k_ax = k;
      break;
    }
  }
  if (k_ax < 0)
    return false;

  double w_c = box.center.dot(e);
  w_lo = w_c - box.half[k_ax];
  w_hi = w_c + box.half[k_ax];

  int k1 = (k_ax + 1) % 3, k2 = (k_ax + 2) % 3;
  auto comp = [](const Position& v, int k) {
    return k == 0 ? v.x : (k == 1 ? v.y : v.z);
  };
  double c1 = comp(box.center, r1), c2 = comp(box.center, r2);
  double a1 = comp(box.axis[k1], r1), a2 = comp(box.axis[k1], r2);
  double b1 = comp(box.axis[k2], r1), b2 = comp(box.axis[k2], r2);
  double h1 = box.half[k1], h2 = box.half[k2];

  px.assign(4, 0.0);
  py.assign(4, 0.0);
  const double sg[4][2] {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  for (int i = 0; i < 4; ++i) {
    px[i] = c1 + sg[i][0] * h1 * a1 + sg[i][1] * h2 * b1;
    py[i] = c2 + sg[i][0] * h1 * a2 + sg[i][1] * h2 * b2;
  }
  if (poly_area(px, py) < 0.0) {
    std::reverse(px.begin(), px.end());
    std::reverse(py.begin(), py.end());
  }
  return true;
}

//==============================================================================
//! Three-valued logic over a postfix region expression.
//==============================================================================
//
// These mirror OP_UNION etc. from cell.h.  They are redefined here so this
// header stays dependency-free and testable; volume_octree.cpp static_asserts
// that they still agree with the real constants.

constexpr int32_t kOpLeftParen {std::numeric_limits<int32_t>::max()};
constexpr int32_t kOpRightParen {std::numeric_limits<int32_t>::max() - 1};
constexpr int32_t kOpComplement {std::numeric_limits<int32_t>::max() - 2};
constexpr int32_t kOpIntersection {std::numeric_limits<int32_t>::max() - 3};
constexpr int32_t kOpUnion {std::numeric_limits<int32_t>::max() - 4};

enum class Tri : uint8_t { kFalse = 0, kTrue = 1, kMaybe = 2 };

inline Tri tri_not(Tri a)
{
  if (a == Tri::kMaybe)
    return Tri::kMaybe;
  return a == Tri::kTrue ? Tri::kFalse : Tri::kTrue;
}

inline Tri tri_and(Tri a, Tri b)
{
  if (a == Tri::kFalse || b == Tri::kFalse)
    return Tri::kFalse;
  if (a == Tri::kTrue && b == Tri::kTrue)
    return Tri::kTrue;
  return Tri::kMaybe;
}

inline Tri tri_or(Tri a, Tri b)
{
  if (a == Tri::kTrue || b == Tri::kTrue)
    return Tri::kTrue;
  if (a == Tri::kFalse && b == Tri::kFalse)
    return Tri::kFalse;
  return Tri::kMaybe;
}

//! Evaluate a region in three-valued logic.
//!
//! \param expr    postfix tokens for a complex region, or the bare
//!                intersection list OpenMC stores for a simple one
//! \param simple  true if expr carries no operators (Region::is_simple())
//! \param sense   callable int32_t -> BoxSense for zero-based surface index
template<typename SenseFn>
Tri evaluate_region_tri(
  const vector<int32_t>& expr, bool simple, SenseFn&& sense)
{
  auto leaf = [&](int32_t token) -> Tri {
    BoxSense bs = sense(std::abs(token) - 1);
    if (bs == BoxSense::BOTH || bs == BoxSense::UNKNOWN)
      return Tri::kMaybe;
    return (bs == BoxSense::POSITIVE) == (token > 0) ? Tri::kTrue : Tri::kFalse;
  };

  if (simple) {
    Tri acc = Tri::kTrue;
    for (int32_t t : expr) {
      acc = tri_and(acc, leaf(t));
      if (acc == Tri::kFalse)
        return Tri::kFalse; // short circuit
    }
    return acc;
  }

  vector<Tri> stack;
  stack.reserve(expr.size());
  for (int32_t token : expr) {
    if (token == kOpUnion) {
      Tri b = stack.back();
      stack.pop_back();
      Tri a = stack.back();
      stack.pop_back();
      stack.push_back(tri_or(a, b));
    } else if (token == kOpIntersection) {
      Tri b = stack.back();
      stack.pop_back();
      Tri a = stack.back();
      stack.pop_back();
      stack.push_back(tri_and(a, b));
    } else if (token == kOpComplement) {
      Tri a = stack.back();
      stack.pop_back();
      stack.push_back(tri_not(a));
    } else {
      stack.push_back(leaf(token));
    }
  }
  return stack.back();
}

} // namespace openmc

#endif // OPENMC_VOLUME_OCTREE_MATH_H
