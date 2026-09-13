#include "openmc/surface.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <set>
#include <utility>

#include <fmt/core.h>

#include "openmc/array.h"
#include "openmc/cell.h"
#include "openmc/container_util.h"
#include "openmc/error.h"
#include "openmc/external/quartic_solver.h"
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/string_utils.h"
#include "openmc/xml_interface.h"

namespace openmc {

//==============================================================================
// Global variables
//==============================================================================

namespace model {
std::unordered_map<int, int> surface_map;
vector<unique_ptr<Surface>> surfaces;
} // namespace model

//==============================================================================
// Helper functions for reading the "coeffs" node of an XML surface element
//==============================================================================

void read_coeffs(
  pugi::xml_node surf_node, int surf_id, std::initializer_list<double*> coeffs)
{
  // Check the given number of coefficients.
  auto coeffs_file = get_node_array<double>(surf_node, "coeffs");
  if (coeffs_file.size() != coeffs.size()) {
    fatal_error(
      fmt::format("Surface {} expects {} coefficient but was given {}", surf_id,
        coeffs.size(), coeffs_file.size()));
  }

  // Copy the coefficients
  int i = 0;
  for (auto c : coeffs) {
    *c = coeffs_file[i++];
  }
}

//==============================================================================
// Surface implementation
//==============================================================================

Surface::Surface() {} // empty constructor

Surface::Surface(pugi::xml_node surf_node)
{
  if (check_for_node(surf_node, "id")) {
    id_ = std::stoi(get_node_value(surf_node, "id"));
    if (contains(settings::source_write_surf_id, id_) ||
        settings::source_write_surf_id.empty()) {
      surf_source_ = true;
    }
  } else {
    fatal_error("Must specify id of surface in geometry XML file.");
  }

  if (check_for_node(surf_node, "name")) {
    name_ = get_node_value(surf_node, "name", false);
  }

  if (check_for_node(surf_node, "boundary")) {
    std::string surf_bc = get_node_value(surf_node, "boundary", true, true);

    if (surf_bc == "transmission" || surf_bc == "transmit" || surf_bc.empty()) {
      // Leave the bc_ a nullptr
    } else if (surf_bc == "vacuum") {
      bc_ = make_unique<VacuumBC>();
    } else if (surf_bc == "reflective" || surf_bc == "reflect" ||
               surf_bc == "reflecting") {
      bc_ = make_unique<ReflectiveBC>();
    } else if (surf_bc == "white") {
      bc_ = make_unique<WhiteBC>();
    } else if (surf_bc == "periodic") {
      // Periodic BCs are handled separately
    } else {
      fatal_error(fmt::format("Unknown boundary condition \"{}\" specified "
                              "on surface {}",
        surf_bc, id_));
    }

    if (check_for_node(surf_node, "albedo") && bc_) {
      double surf_alb = std::stod(get_node_value(surf_node, "albedo"));

      if (surf_alb < 0.0)
        fatal_error(fmt::format("Surface {} has an albedo of {}. "
                                "Albedo values must be positive.",
          id_, surf_alb));

      if (surf_alb > 1.0)
        warning(fmt::format("Surface {} has an albedo of {}. "
                            "Albedos greater than 1 may cause "
                            "unphysical behaviour.",
          id_, surf_alb));

      bc_->set_albedo(surf_alb);
    }
  }
}

bool Surface::sense(Position r, Direction u) const
{
  // Evaluate the surface equation at the particle's coordinates to determine
  // which side the particle is on.
  const double f = evaluate(r);

  // Check which side of surface the point is on.
  if (std::abs(f) < FP_COINCIDENT) {
    // Particle may be coincident with this surface. To determine the sense, we
    // look at the direction of the particle relative to the surface normal (by
    // default in the positive direction) via their dot product.
    return u.dot(normal(r)) > 0.0;
  }
  return f > 0.0;
}

Direction Surface::reflect(Position r, Direction u, GeometryState* p) const
{
  // Determine projection of direction onto normal and squared magnitude of
  // normal.
  Direction n = normal(r);

  // Reflect direction according to normal.
  return u.reflect(n);
}

Direction Surface::diffuse_reflect(
  Position r, Direction u, uint64_t* seed) const
{
  // Diffuse reflect direction according to the normal.
  // cosine distribution

  Direction n = this->normal(r);
  n /= n.norm();
  const double projection = n.dot(u);

  // sample from inverse function, u=sqrt(rand) since p(u)=2u, so F(u)=u^2
  const double mu =
    (projection >= 0.0) ? -std::sqrt(prn(seed)) : std::sqrt(prn(seed));

  // sample azimuthal distribution uniformly
  u = rotate_angle(n, mu, nullptr, seed);

  // normalize the direction
  return u / u.norm();
}

void Surface::to_hdf5(hid_t group_id) const
{
  hid_t surf_group = create_group(group_id, fmt::format("surface {}", id_));

  if (geom_type() == GeometryType::DAG) {
    write_string(surf_group, "geom_type", "dagmc", false);
  } else if (geom_type() == GeometryType::CSG) {
    write_string(surf_group, "geom_type", "csg", false);

    if (bc_) {
      write_string(surf_group, "boundary_type", bc_->type(), false);
      bc_->to_hdf5(surf_group);

      // write periodic surface ID
      if (bc_->type() == "periodic") {
        auto pbc = dynamic_cast<PeriodicBC*>(bc_.get());
        Surface& surf1 {*model::surfaces[pbc->i_surf()]};
        Surface& surf2 {*model::surfaces[pbc->j_surf()]};

        if (id_ == surf1.id_) {
          write_dataset(surf_group, "periodic_surface_id", surf2.id_);
        } else {
          write_dataset(surf_group, "periodic_surface_id", surf1.id_);
        }
      }
    } else {
      write_string(surf_group, "boundary_type", "transmission", false);
    }
  }

  if (!name_.empty()) {
    write_string(surf_group, "name", name_, false);
  }

  to_hdf5_inner(surf_group);

  close_group(surf_group);
}

//==============================================================================
// Generic functions for x-, y-, and z-, planes.
//==============================================================================

// The template parameter indicates the axis normal to the plane.
template<int i>
double axis_aligned_plane_distance(
  Position r, Direction u, bool coincident, double offset)
{
  const double f = offset - r[i];
  if (coincident || std::abs(f) < FP_COINCIDENT || u[i] == 0.0)
    return INFTY;
  const double d = f / u[i];
  if (d < 0.0)
    return INFTY;
  return d;
}

//==============================================================================
// SurfaceXPlane implementation
//==============================================================================

SurfaceXPlane::SurfaceXPlane(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_});
}

double SurfaceXPlane::evaluate(Position r) const
{
  return r.x - x0_;
}

double SurfaceXPlane::distance(Position r, Direction u, bool coincident) const
{
  return axis_aligned_plane_distance<0>(r, u, coincident, x0_);
}

Direction SurfaceXPlane::normal(Position r) const
{
  return {1., 0., 0.};
}

void SurfaceXPlane::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "x-plane", false);
  array<double, 1> coeffs {{x0_}};
  write_dataset(group_id, "coefficients", coeffs);
}

BoundingBox SurfaceXPlane::bounding_box(bool pos_side) const
{
  if (pos_side) {
    return {{x0_, -INFTY, -INFTY}, {INFTY, INFTY, INFTY}};
  } else {
    return {{-INFTY, -INFTY, -INFTY}, {x0_, INFTY, INFTY}};
  }
}

//==============================================================================
// SurfaceYPlane implementation
//==============================================================================

SurfaceYPlane::SurfaceYPlane(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&y0_});
}

double SurfaceYPlane::evaluate(Position r) const
{
  return r.y - y0_;
}

double SurfaceYPlane::distance(Position r, Direction u, bool coincident) const
{
  return axis_aligned_plane_distance<1>(r, u, coincident, y0_);
}

Direction SurfaceYPlane::normal(Position r) const
{
  return {0., 1., 0.};
}

void SurfaceYPlane::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "y-plane", false);
  array<double, 1> coeffs {{y0_}};
  write_dataset(group_id, "coefficients", coeffs);
}

BoundingBox SurfaceYPlane::bounding_box(bool pos_side) const
{
  if (pos_side) {
    return {{-INFTY, y0_, -INFTY}, {INFTY, INFTY, INFTY}};
  } else {
    return {{-INFTY, -INFTY, -INFTY}, {INFTY, y0_, INFTY}};
  }
}

//==============================================================================
// SurfaceZPlane implementation
//==============================================================================

SurfaceZPlane::SurfaceZPlane(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&z0_});
}

double SurfaceZPlane::evaluate(Position r) const
{
  return r.z - z0_;
}

double SurfaceZPlane::distance(Position r, Direction u, bool coincident) const
{
  return axis_aligned_plane_distance<2>(r, u, coincident, z0_);
}

Direction SurfaceZPlane::normal(Position r) const
{
  return {0., 0., 1.};
}

void SurfaceZPlane::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "z-plane", false);
  array<double, 1> coeffs {{z0_}};
  write_dataset(group_id, "coefficients", coeffs);
}

BoundingBox SurfaceZPlane::bounding_box(bool pos_side) const
{
  if (pos_side) {
    return {{-INFTY, -INFTY, z0_}, {INFTY, INFTY, INFTY}};
  } else {
    return {{-INFTY, -INFTY, -INFTY}, {INFTY, INFTY, z0_}};
  }
}

//==============================================================================
// SurfacePlane implementation
//==============================================================================

SurfacePlane::SurfacePlane(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&A_, &B_, &C_, &D_});
}

double SurfacePlane::evaluate(Position r) const
{
  return A_ * r.x + B_ * r.y + C_ * r.z - D_;
}

BoundingBox SurfacePlane::bounding_box(bool pos_side) const
{
  // A general plane bounds a half-space in one direction only when its normal
  // is parallel to a coordinate axis; otherwise both half-spaces are unbounded
  // along every axis. This mirrors PlaneMixin.bounding_box on the Python side,
  // so that a plane whose off-axis coefficients are rotation-matrix roundoff
  // (e.g. B = 1 with A = C = 6.1e-17) yields the same box through both APIs.
  const Direction n = normal({});
  const double norm = n.norm();
  if (norm == 0.0)
    return {};

  int axis = -1;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(std::abs(n[i] / norm) - 1.0) > PLANE_ALIGNMENT_TOL)
      continue;

    bool aligned = true;
    for (int j = 0; j < 3; ++j) {
      if (j != i && std::abs(n[j] / norm) > PLANE_ALIGNMENT_TOL) {
        aligned = false;
        break;
      }
    }
    if (aligned) {
      axis = i;
      break;
    }
  }
  if (axis == -1)
    return {};

  // The half-space is bounded below when the outward normal points along the
  // positive axis direction and we are on the positive side, or vice versa.
  BoundingBox bbox;
  const double intercept = D_ / n[axis];
  if (pos_side == (n[axis] > 0.0)) {
    bbox.min[axis] = intercept;
  } else {
    bbox.max[axis] = intercept;
  }
  return bbox;
}

double SurfacePlane::distance(Position r, Direction u, bool coincident) const
{
  const double f = A_ * r.x + B_ * r.y + C_ * r.z - D_;
  const double projection = A_ * u.x + B_ * u.y + C_ * u.z;
  if (coincident || std::abs(f) < FP_COINCIDENT || projection == 0.0) {
    return INFTY;
  } else {
    const double d = -f / projection;
    if (d < 0.0)
      return INFTY;
    return d;
  }
}

Direction SurfacePlane::normal(Position r) const
{
  return {A_, B_, C_};
}

void SurfacePlane::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "plane", false);
  array<double, 4> coeffs {{A_, B_, C_, D_}};
  write_dataset(group_id, "coefficients", coeffs);
}

//==============================================================================
// Generic functions for x-, y-, and z-, cylinders
//==============================================================================

// The template parameters indicate the axes perpendicular to the axis of the
// cylinder.  offset1 and offset2 should correspond with i1 and i2,
// respectively.
template<int i1, int i2>
double axis_aligned_cylinder_evaluate(
  Position r, double offset1, double offset2, double radius)
{
  const double r1 = r.get<i1>() - offset1;
  const double r2 = r.get<i2>() - offset2;
  return r1 * r1 + r2 * r2 - radius * radius;
}

// The first template parameter indicates which axis the cylinder is aligned to.
// The other two parameters indicate the other two axes.  offset1 and offset2
// should correspond with i2 and i3, respectively.
template<int i1, int i2, int i3>
double axis_aligned_cylinder_distance(Position r, Direction u, bool coincident,
  double offset1, double offset2, double radius)
{
  const double a = 1.0 - u.get<i1>() * u.get<i1>(); // u^2 + v^2
  if (a == 0.0)
    return INFTY;

  const double r2 = r.get<i2>() - offset1;
  const double r3 = r.get<i3>() - offset2;
  const double k = r2 * u.get<i2>() + r3 * u.get<i3>();
  const double c = r2 * r2 + r3 * r3 - radius * radius;
  const double quad = k * k - a * c;

  if (quad < 0.0) {
    // No intersection with cylinder.
    return INFTY;

  } else if (coincident || std::abs(c) < FP_COINCIDENT) {
    // Particle is on the cylinder, thus one distance is positive/negative
    // and the other is zero. The sign of k determines if we are facing in or
    // out.
    if (k >= 0.0) {
      return INFTY;
    } else {
      return (-k + sqrt(quad)) / a;
    }

  } else if (c < 0.0) {
    // Particle is inside the cylinder, thus one distance must be negative
    // and one must be positive. The positive distance will be the one with
    // negative sign on sqrt(quad).
    return (-k + sqrt(quad)) / a;

  } else {
    // Particle is outside the cylinder, thus both distances are either
    // positive or negative. If positive, the smaller distance is the one
    // with positive sign on sqrt(quad).
    const double d = (-k - sqrt(quad)) / a;
    if (d < 0.0)
      return INFTY;
    return d;
  }
}

// The first template parameter indicates which axis the cylinder is aligned to.
// The other two parameters indicate the other two axes.  offset1 and offset2
// should correspond with i2 and i3, respectively.
template<int i1, int i2, int i3>
Direction axis_aligned_cylinder_normal(
  Position r, double offset1, double offset2)
{
  Direction u;
  u.get<i2>() = 2.0 * (r.get<i2>() - offset1);
  u.get<i3>() = 2.0 * (r.get<i3>() - offset2);
  u.get<i1>() = 0.0;
  return u;
}

//==============================================================================
// SurfaceXCylinder implementation
//==============================================================================

SurfaceXCylinder::SurfaceXCylinder(pugi::xml_node surf_node)
  : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&y0_, &z0_, &radius_});
}

double SurfaceXCylinder::evaluate(Position r) const
{
  return axis_aligned_cylinder_evaluate<1, 2>(r, y0_, z0_, radius_);
}

double SurfaceXCylinder::distance(
  Position r, Direction u, bool coincident) const
{
  return axis_aligned_cylinder_distance<0, 1, 2>(
    r, u, coincident, y0_, z0_, radius_);
}

Direction SurfaceXCylinder::normal(Position r) const
{
  return axis_aligned_cylinder_normal<0, 1, 2>(r, y0_, z0_);
}

void SurfaceXCylinder::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "x-cylinder", false);
  array<double, 3> coeffs {{y0_, z0_, radius_}};
  write_dataset(group_id, "coefficients", coeffs);
}

BoundingBox SurfaceXCylinder::bounding_box(bool pos_side) const
{
  if (!pos_side) {
    return {{-INFTY, y0_ - radius_, z0_ - radius_},
      {INFTY, y0_ + radius_, z0_ + radius_}};
  } else {
    return {};
  }
}
//==============================================================================
// SurfaceYCylinder implementation
//==============================================================================

SurfaceYCylinder::SurfaceYCylinder(pugi::xml_node surf_node)
  : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &z0_, &radius_});
}

double SurfaceYCylinder::evaluate(Position r) const
{
  return axis_aligned_cylinder_evaluate<0, 2>(r, x0_, z0_, radius_);
}

double SurfaceYCylinder::distance(
  Position r, Direction u, bool coincident) const
{
  return axis_aligned_cylinder_distance<1, 0, 2>(
    r, u, coincident, x0_, z0_, radius_);
}

Direction SurfaceYCylinder::normal(Position r) const
{
  return axis_aligned_cylinder_normal<1, 0, 2>(r, x0_, z0_);
}

void SurfaceYCylinder::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "y-cylinder", false);
  array<double, 3> coeffs {{x0_, z0_, radius_}};
  write_dataset(group_id, "coefficients", coeffs);
}

BoundingBox SurfaceYCylinder::bounding_box(bool pos_side) const
{
  if (!pos_side) {
    return {{x0_ - radius_, -INFTY, z0_ - radius_},
      {x0_ + radius_, INFTY, z0_ + radius_}};
  } else {
    return {};
  }
}

//==============================================================================
// SurfaceZCylinder implementation
//==============================================================================

SurfaceZCylinder::SurfaceZCylinder(pugi::xml_node surf_node)
  : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &radius_});
}

double SurfaceZCylinder::evaluate(Position r) const
{
  return axis_aligned_cylinder_evaluate<0, 1>(r, x0_, y0_, radius_);
}

double SurfaceZCylinder::distance(
  Position r, Direction u, bool coincident) const
{
  return axis_aligned_cylinder_distance<2, 0, 1>(
    r, u, coincident, x0_, y0_, radius_);
}

Direction SurfaceZCylinder::normal(Position r) const
{
  return axis_aligned_cylinder_normal<2, 0, 1>(r, x0_, y0_);
}

void SurfaceZCylinder::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "z-cylinder", false);
  array<double, 3> coeffs {{x0_, y0_, radius_}};
  write_dataset(group_id, "coefficients", coeffs);
}

BoundingBox SurfaceZCylinder::bounding_box(bool pos_side) const
{
  if (!pos_side) {
    return {{x0_ - radius_, y0_ - radius_, -INFTY},
      {x0_ + radius_, y0_ + radius_, INFTY}};
  } else {
    return {};
  }
}

//==============================================================================
// SurfaceSphere implementation
//==============================================================================

SurfaceSphere::SurfaceSphere(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &z0_, &radius_});
}

double SurfaceSphere::evaluate(Position r) const
{
  const double x = r.x - x0_;
  const double y = r.y - y0_;
  const double z = r.z - z0_;
  return x * x + y * y + z * z - radius_ * radius_;
}

double SurfaceSphere::distance(Position r, Direction u, bool coincident) const
{
  const double x = r.x - x0_;
  const double y = r.y - y0_;
  const double z = r.z - z0_;
  const double k = x * u.x + y * u.y + z * u.z;
  const double c = x * x + y * y + z * z - radius_ * radius_;
  const double quad = k * k - c;

  if (quad < 0.0) {
    // No intersection with sphere.
    return INFTY;

  } else if (coincident || std::abs(c) < FP_COINCIDENT) {
    // Particle is on the sphere, thus one distance is positive/negative and
    // the other is zero. The sign of k determines if we are facing in or out.
    if (k >= 0.0) {
      return INFTY;
    } else {
      return -k + sqrt(quad);
    }

  } else if (c < 0.0) {
    // Particle is inside the sphere, thus one distance must be negative and
    // one must be positive. The positive distance will be the one with
    // negative sign on sqrt(quad)
    return -k + sqrt(quad);

  } else {
    // Particle is outside the sphere, thus both distances are either positive
    // or negative. If positive, the smaller distance is the one with positive
    // sign on sqrt(quad).
    const double d = -k - sqrt(quad);
    if (d < 0.0)
      return INFTY;
    return d;
  }
}

Direction SurfaceSphere::normal(Position r) const
{
  return {2.0 * (r.x - x0_), 2.0 * (r.y - y0_), 2.0 * (r.z - z0_)};
}

void SurfaceSphere::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "sphere", false);
  array<double, 4> coeffs {{x0_, y0_, z0_, radius_}};
  write_dataset(group_id, "coefficients", coeffs);
}

BoundingBox SurfaceSphere::bounding_box(bool pos_side) const
{
  if (!pos_side) {
    return {{x0_ - radius_, y0_ - radius_, z0_ - radius_},
      {x0_ + radius_, y0_ + radius_, z0_ + radius_}};
  } else {
    return {};
  }
}

//==============================================================================
// Generic functions for x-, y-, and z-, cones
//==============================================================================

// The first template parameter indicates which axis the cone is aligned to.
// The other two parameters indicate the other two axes.  offset1, offset2,
// and offset3 should correspond with i1, i2, and i3, respectively.
template<int i1, int i2, int i3>
double axis_aligned_cone_evaluate(
  Position r, double offset1, double offset2, double offset3, double radius_sq)
{
  const double r1 = r.get<i1>() - offset1;
  const double r2 = r.get<i2>() - offset2;
  const double r3 = r.get<i3>() - offset3;
  return r2 * r2 + r3 * r3 - radius_sq * r1 * r1;
}

// The first template parameter indicates which axis the cone is aligned to.
// The other two parameters indicate the other two axes.  offset1, offset2,
// and offset3 should correspond with i1, i2, and i3, respectively.
template<int i1, int i2, int i3>
double axis_aligned_cone_distance(Position r, Direction u, bool coincident,
  double offset1, double offset2, double offset3, double radius_sq)
{
  const double r1 = r.get<i1>() - offset1;
  const double r2 = r.get<i2>() - offset2;
  const double r3 = r.get<i3>() - offset3;
  const double a = u.get<i2>() * u.get<i2>() + u.get<i3>() * u.get<i3>() -
                   radius_sq * u.get<i1>() * u.get<i1>();
  const double k =
    r2 * u.get<i2>() + r3 * u.get<i3>() - radius_sq * r1 * u.get<i1>();
  const double c = r2 * r2 + r3 * r3 - radius_sq * r1 * r1;
  double quad = k * k - a * c;

  double d;

  if (quad < 0.0) {
    // No intersection with cone.
    return INFTY;

  } else if (coincident || std::abs(c) < FP_COINCIDENT) {
    // Particle is on the cone, thus one distance is positive/negative
    // and the other is zero. The sign of k determines if we are facing in or
    // out.
    if (k >= 0.0) {
      d = (-k - sqrt(quad)) / a;
    } else {
      d = (-k + sqrt(quad)) / a;
    }

  } else {
    // Calculate both solutions to the quadratic.
    quad = sqrt(quad);
    d = (-k - quad) / a;
    const double b = (-k + quad) / a;

    // Determine the smallest positive solution.
    if (d < 0.0) {
      if (b > 0.0)
        d = b;
    } else {
      if (b > 0.0) {
        if (b < d)
          d = b;
      }
    }
  }

  // If the distance was negative, set boundary distance to infinity.
  if (d <= 0.0)
    return INFTY;
  return d;
}

// The first template parameter indicates which axis the cone is aligned to.
// The other two parameters indicate the other two axes.  offset1, offset2,
// and offset3 should correspond with i1, i2, and i3, respectively.
template<int i1, int i2, int i3>
Direction axis_aligned_cone_normal(
  Position r, double offset1, double offset2, double offset3, double radius_sq)
{
  Direction u;
  u.get<i1>() = -2.0 * radius_sq * (r.get<i1>() - offset1);
  u.get<i2>() = 2.0 * (r.get<i2>() - offset2);
  u.get<i3>() = 2.0 * (r.get<i3>() - offset3);
  return u;
}

//==============================================================================
// SurfaceXCone implementation
//==============================================================================

SurfaceXCone::SurfaceXCone(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &z0_, &radius_sq_});
}

double SurfaceXCone::evaluate(Position r) const
{
  return axis_aligned_cone_evaluate<0, 1, 2>(r, x0_, y0_, z0_, radius_sq_);
}

double SurfaceXCone::distance(Position r, Direction u, bool coincident) const
{
  return axis_aligned_cone_distance<0, 1, 2>(
    r, u, coincident, x0_, y0_, z0_, radius_sq_);
}

Direction SurfaceXCone::normal(Position r) const
{
  return axis_aligned_cone_normal<0, 1, 2>(r, x0_, y0_, z0_, radius_sq_);
}

void SurfaceXCone::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "x-cone", false);
  array<double, 4> coeffs {{x0_, y0_, z0_, radius_sq_}};
  write_dataset(group_id, "coefficients", coeffs);
}

//==============================================================================
// SurfaceYCone implementation
//==============================================================================

SurfaceYCone::SurfaceYCone(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &z0_, &radius_sq_});
}

double SurfaceYCone::evaluate(Position r) const
{
  return axis_aligned_cone_evaluate<1, 0, 2>(r, y0_, x0_, z0_, radius_sq_);
}

double SurfaceYCone::distance(Position r, Direction u, bool coincident) const
{
  return axis_aligned_cone_distance<1, 0, 2>(
    r, u, coincident, y0_, x0_, z0_, radius_sq_);
}

Direction SurfaceYCone::normal(Position r) const
{
  return axis_aligned_cone_normal<1, 0, 2>(r, y0_, x0_, z0_, radius_sq_);
}

void SurfaceYCone::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "y-cone", false);
  array<double, 4> coeffs {{x0_, y0_, z0_, radius_sq_}};
  write_dataset(group_id, "coefficients", coeffs);
}

//==============================================================================
// SurfaceZCone implementation
//==============================================================================

SurfaceZCone::SurfaceZCone(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &z0_, &radius_sq_});
}

double SurfaceZCone::evaluate(Position r) const
{
  return axis_aligned_cone_evaluate<2, 0, 1>(r, z0_, x0_, y0_, radius_sq_);
}

double SurfaceZCone::distance(Position r, Direction u, bool coincident) const
{
  return axis_aligned_cone_distance<2, 0, 1>(
    r, u, coincident, z0_, x0_, y0_, radius_sq_);
}

Direction SurfaceZCone::normal(Position r) const
{
  return axis_aligned_cone_normal<2, 0, 1>(r, z0_, x0_, y0_, radius_sq_);
}

void SurfaceZCone::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "z-cone", false);
  array<double, 4> coeffs {{x0_, y0_, z0_, radius_sq_}};
  write_dataset(group_id, "coefficients", coeffs);
}

//==============================================================================
// SurfaceQuadric implementation
//==============================================================================

SurfaceQuadric::SurfaceQuadric(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(
    surf_node, id_, {&A_, &B_, &C_, &D_, &E_, &F_, &G_, &H_, &J_, &K_});
}

double SurfaceQuadric::evaluate(Position r) const
{
  const double x = r.x;
  const double y = r.y;
  const double z = r.z;
  return x * (A_ * x + D_ * y + G_) + y * (B_ * y + E_ * z + H_) +
         z * (C_ * z + F_ * x + J_) + K_;
}

double SurfaceQuadric::distance(
  Position r, Direction ang, bool coincident) const
{
  const double& x = r.x;
  const double& y = r.y;
  const double& z = r.z;
  const double& u = ang.x;
  const double& v = ang.y;
  const double& w = ang.z;

  const double a =
    A_ * u * u + B_ * v * v + C_ * w * w + D_ * u * v + E_ * v * w + F_ * u * w;
  const double k = A_ * u * x + B_ * v * y + C_ * w * z +
                   0.5 * (D_ * (u * y + v * x) + E_ * (v * z + w * y) +
                           F_ * (w * x + u * z) + G_ * u + H_ * v + J_ * w);
  const double c = A_ * x * x + B_ * y * y + C_ * z * z + D_ * x * y +
                   E_ * y * z + F_ * x * z + G_ * x + H_ * y + J_ * z + K_;
  double quad = k * k - a * c;

  double d;

  if (quad < 0.0) {
    // No intersection with surface.
    return INFTY;

  } else if (coincident || std::abs(c) < FP_COINCIDENT) {
    // Particle is on the surface, thus one distance is positive/negative and
    // the other is zero. The sign of k determines which distance is zero and
    // which is not. Additionally, if a is zero, it means the particle is on
    // a plane-like surface.
    if (a == 0.0) {
      d = INFTY; // see the below explanation
    } else if (k >= 0.0) {
      d = (-k - sqrt(quad)) / a;
    } else {
      d = (-k + sqrt(quad)) / a;
    }

  } else if (a == 0.0) {
    // Given the orientation of the particle, the quadric looks like a plane in
    // this case, and thus we have only one solution despite potentially having
    // quad > 0.0. While the term under the square root may be real, in one
    // case of the +/- of the quadratic formula, 0/0 results, and in another, a
    // finite value over 0 results. Applying L'Hopital's to the 0/0 case gives
    // the below. Alternatively this can be found by simply putting a=0 in the
    // equation ax^2 + bx + c = 0.
    d = -0.5 * c / k;
  } else {
    // Calculate both solutions to the quadratic.
    quad = sqrt(quad);
    d = (-k - quad) / a;
    double b = (-k + quad) / a;

    // Determine the smallest positive solution.
    if (d < 0.0) {
      if (b > 0.0)
        d = b;
    } else {
      if (b > 0.0) {
        if (b < d)
          d = b;
      }
    }
  }

  // If the distance was negative, set boundary distance to infinity.
  if (d <= 0.0)
    return INFTY;
  return d;
}

Direction SurfaceQuadric::normal(Position r) const
{
  const double& x = r.x;
  const double& y = r.y;
  const double& z = r.z;
  return {2.0 * A_ * x + D_ * y + F_ * z + G_,
    2.0 * B_ * y + D_ * x + E_ * z + H_, 2.0 * C_ * z + E_ * y + F_ * x + J_};
}

void SurfaceQuadric::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "quadric", false);
  array<double, 10> coeffs {{A_, B_, C_, D_, E_, F_, G_, H_, J_, K_}};
  write_dataset(group_id, "coefficients", coeffs);
}

//==============================================================================
// Torus helper functions
//==============================================================================

double torus_distance(double x1, double x2, double x3, double u1, double u2,
  double u3, double A, double B, double C, bool coincident)
{
  // Coefficients for equation: (c2 t^2 + c1 t + c0)^2 = c2' t^2 + c1' t + c0'
  double D = (C * C) / (B * B);
  double c2 = u1 * u1 + u2 * u2 + D * u3 * u3;
  double c1 = 2 * (u1 * x1 + u2 * x2 + D * u3 * x3);
  double c0 = x1 * x1 + x2 * x2 + D * x3 * x3 + A * A - C * C;
  double four_A2 = 4 * A * A;
  double c2p = four_A2 * (u1 * u1 + u2 * u2);
  double c1p = 2 * four_A2 * (u1 * x1 + u2 * x2);
  double c0p = four_A2 * (x1 * x1 + x2 * x2);

  // Coefficient for equation: a t^4 + b t^3 + c t^2 + d t + e = 0. If the point
  // is coincident, the 'e' coefficient should be zero. Explicitly setting it to
  // zero helps avoid numerical issues below with root finding.
  double coeff[5];
  coeff[0] = coincident ? 0.0 : c0 * c0 - c0p;
  coeff[1] = 2 * c0 * c1 - c1p;
  coeff[2] = c1 * c1 + 2 * c0 * c2 - c2p;
  coeff[3] = 2 * c1 * c2;
  coeff[4] = c2 * c2;

  std::complex<double> roots[4];
  oqs::quartic_solver(coeff, roots);

  // Find smallest positive, real root. In the case where the particle is
  // coincident with the surface, we are sure to have one root very close to
  // zero but possibly small and positive. A tolerance is set to discard that
  // zero.
  double distance = INFTY;
  double cutoff = coincident ? TORUS_TOL : 0.0;
  for (int i = 0; i < 4; ++i) {
    if (roots[i].imag() == 0) {
      double root = roots[i].real();
      if (root > cutoff && root < distance) {
        // Avoid roots corresponding to internal surfaces
        double s1 = x1 + u1 * root;
        double s2 = x2 + u2 * root;
        double s3 = x3 + u3 * root;
        double check = D * s3 * s3 + s1 * s1 + s2 * s2 + A * A - C * C;
        if (check >= 0) {
          distance = root;
        }
      }
    }
  }
  return distance;
}

//==============================================================================
// SurfaceXTorus implementation
//==============================================================================

SurfaceXTorus::SurfaceXTorus(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &z0_, &A_, &B_, &C_});
}

void SurfaceXTorus::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "x-torus", false);
  std::array<double, 6> coeffs {{x0_, y0_, z0_, A_, B_, C_}};
  write_dataset(group_id, "coefficients", coeffs);
}

double SurfaceXTorus::evaluate(Position r) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;
  return (x * x) / (B_ * B_) +
         std::pow(std::sqrt(y * y + z * z) - A_, 2) / (C_ * C_) - 1.;
}

BoundingBox SurfaceXTorus::bounding_box(bool pos_side) const
{
  // The torus interior is compact: it extends +/-B_ along the axis of
  // revolution and +/-(A_ + C_) in the two perpendicular directions. Mirrors
  // XTorus.bounding_box on the Python side.
  if (pos_side)
    return {};
  return {{x0_ - B_, y0_ - A_ - C_, z0_ - A_ - C_},
    {x0_ + B_, y0_ + A_ + C_, z0_ + A_ + C_}};
}

double SurfaceXTorus::distance(Position r, Direction u, bool coincident) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;
  return torus_distance(y, z, x, u.y, u.z, u.x, A_, B_, C_, coincident);
}

Direction SurfaceXTorus::normal(Position r) const
{
  // reduce the expansion of the full form for torus
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;

  // f(x,y,z) = x^2/B^2 + (sqrt(y^2 + z^2) - A)^2/C^2 - 1
  // ∂f/∂x = 2x/B^2
  // ∂f/∂y = 2y(g - A)/(g*C^2) where g = sqrt(y^2 + z^2)
  // ∂f/∂z = 2z(g - A)/(g*C^2)
  // Multiplying by g*C^2*B^2 / 2 gives:
  double g = std::sqrt(y * y + z * z);
  double nx = C_ * C_ * g * x;
  double ny = y * (g - A_) * B_ * B_;
  double nz = z * (g - A_) * B_ * B_;
  Direction n(nx, ny, nz);
  return n / n.norm();
}

//==============================================================================
// SurfaceYTorus implementation
//==============================================================================

SurfaceYTorus::SurfaceYTorus(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &z0_, &A_, &B_, &C_});
}

void SurfaceYTorus::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "y-torus", false);
  std::array<double, 6> coeffs {{x0_, y0_, z0_, A_, B_, C_}};
  write_dataset(group_id, "coefficients", coeffs);
}

double SurfaceYTorus::evaluate(Position r) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;
  return (y * y) / (B_ * B_) +
         std::pow(std::sqrt(x * x + z * z) - A_, 2) / (C_ * C_) - 1.;
}

BoundingBox SurfaceYTorus::bounding_box(bool pos_side) const
{
  if (pos_side)
    return {};
  return {{x0_ - A_ - C_, y0_ - B_, z0_ - A_ - C_},
    {x0_ + A_ + C_, y0_ + B_, z0_ + A_ + C_}};
}

double SurfaceYTorus::distance(Position r, Direction u, bool coincident) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;
  return torus_distance(x, z, y, u.x, u.z, u.y, A_, B_, C_, coincident);
}

Direction SurfaceYTorus::normal(Position r) const
{
  // reduce the expansion of the full form for torus
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;

  // f(x,y,z) = y^2/B^2 + (sqrt(x^2 + z^2) - A)^2/C^2 - 1
  // ∂f/∂x = 2x(g - A)/(g*C^2) where g = sqrt(x^2 + z^2)
  // ∂f/∂y = 2y/B^2
  // ∂f/∂z = 2z(g - A)/(g*C^2)
  // Multiplying by g*C^2*B^2 / 2 gives:
  double g = std::sqrt(x * x + z * z);
  double nx = x * (g - A_) * B_ * B_;
  double ny = C_ * C_ * g * y;
  double nz = z * (g - A_) * B_ * B_;
  Direction n(nx, ny, nz);
  return n / n.norm();
}

//==============================================================================
// SurfaceZTorus implementation
//==============================================================================

SurfaceZTorus::SurfaceZTorus(pugi::xml_node surf_node) : Surface(surf_node)
{
  read_coeffs(surf_node, id_, {&x0_, &y0_, &z0_, &A_, &B_, &C_});
}

void SurfaceZTorus::to_hdf5_inner(hid_t group_id) const
{
  write_string(group_id, "type", "z-torus", false);
  std::array<double, 6> coeffs {{x0_, y0_, z0_, A_, B_, C_}};
  write_dataset(group_id, "coefficients", coeffs);
}

double SurfaceZTorus::evaluate(Position r) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;
  return (z * z) / (B_ * B_) +
         std::pow(std::sqrt(x * x + y * y) - A_, 2) / (C_ * C_) - 1.;
}

BoundingBox SurfaceZTorus::bounding_box(bool pos_side) const
{
  if (pos_side)
    return {};
  return {{x0_ - A_ - C_, y0_ - A_ - C_, z0_ - B_},
    {x0_ + A_ + C_, y0_ + A_ + C_, z0_ + B_}};
}

double SurfaceZTorus::distance(Position r, Direction u, bool coincident) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;
  return torus_distance(x, y, z, u.x, u.y, u.z, A_, B_, C_, coincident);
}

Direction SurfaceZTorus::normal(Position r) const
{
  // reduce the expansion of the full form for torus
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;

  // f(x,y,z) = z^2/B^2 + (sqrt(x^2 + y^2) - A)^2/C^2 - 1
  // ∂f/∂x = 2x(g - A)/(g*C^2) where g = sqrt(x^2 + y^2)
  // ∂f/∂y = 2y(g - A)/(g*C^2)
  // ∂f/∂z = 2z/B^2
  // Multiplying by g*C^2*B^2 / 2 gives:
  double g = std::sqrt(x * x + y * y);
  double nx = x * (g - A_) * B_ * B_;
  double ny = y * (g - A_) * B_ * B_;
  double nz = C_ * C_ * g * z;
  Position n(nx, ny, nz);
  return n / n.norm();
}

//==============================================================================

void read_surfaces(pugi::xml_node node,
  std::set<std::pair<int, int>>& periodic_pairs,
  std::unordered_map<int, double>& albedo_map,
  std::unordered_map<int, int>& periodic_sense_map)
{
  // Count the number of surfaces
  int n_surfaces = 0;
  for (pugi::xml_node surf_node : node.children("surface")) {
    n_surfaces++;
  }

  // Loop over XML surface elements and populate the array.  Keep track of
  // periodic surfaces and their albedos.
  model::surfaces.reserve(n_surfaces);
  {
    pugi::xml_node surf_node;
    int i_surf;
    for (surf_node = node.child("surface"), i_surf = 0; surf_node;
         surf_node = surf_node.next_sibling("surface"), i_surf++) {
      std::string surf_type = get_node_value(surf_node, "type", true, true);

      // Allocate and initialize the new surface

      if (surf_type == "x-plane") {
        model::surfaces.push_back(make_unique<SurfaceXPlane>(surf_node));

      } else if (surf_type == "y-plane") {
        model::surfaces.push_back(make_unique<SurfaceYPlane>(surf_node));

      } else if (surf_type == "z-plane") {
        model::surfaces.push_back(make_unique<SurfaceZPlane>(surf_node));

      } else if (surf_type == "plane") {
        model::surfaces.push_back(make_unique<SurfacePlane>(surf_node));

      } else if (surf_type == "x-cylinder") {
        model::surfaces.push_back(make_unique<SurfaceXCylinder>(surf_node));

      } else if (surf_type == "y-cylinder") {
        model::surfaces.push_back(make_unique<SurfaceYCylinder>(surf_node));

      } else if (surf_type == "z-cylinder") {
        model::surfaces.push_back(make_unique<SurfaceZCylinder>(surf_node));

      } else if (surf_type == "sphere") {
        model::surfaces.push_back(make_unique<SurfaceSphere>(surf_node));

      } else if (surf_type == "x-cone") {
        model::surfaces.push_back(make_unique<SurfaceXCone>(surf_node));

      } else if (surf_type == "y-cone") {
        model::surfaces.push_back(make_unique<SurfaceYCone>(surf_node));

      } else if (surf_type == "z-cone") {
        model::surfaces.push_back(make_unique<SurfaceZCone>(surf_node));

      } else if (surf_type == "quadric") {
        model::surfaces.push_back(make_unique<SurfaceQuadric>(surf_node));

      } else if (surf_type == "x-torus") {
        model::surfaces.push_back(std::make_unique<SurfaceXTorus>(surf_node));

      } else if (surf_type == "y-torus") {
        model::surfaces.push_back(std::make_unique<SurfaceYTorus>(surf_node));

      } else if (surf_type == "z-torus") {
        model::surfaces.push_back(std::make_unique<SurfaceZTorus>(surf_node));

      } else {
        fatal_error(fmt::format("Invalid surface type, \"{}\"", surf_type));
      }

      // Check for a periodic surface
      if (check_for_node(surf_node, "boundary")) {
        std::string surf_bc = get_node_value(surf_node, "boundary", true, true);
        if (surf_bc == "periodic") {
          periodic_sense_map[model::surfaces.back()->id_] = 0;
          // Check for surface albedo. Skip sanity check as it is already done
          // in the Surface class's constructor.
          if (check_for_node(surf_node, "albedo")) {
            albedo_map[model::surfaces.back()->id_] =
              std::stod(get_node_value(surf_node, "albedo"));
          }
          if (check_for_node(surf_node, "periodic_surface_id")) {
            int i_periodic =
              std::stoi(get_node_value(surf_node, "periodic_surface_id"));
            int lo_id = std::min(model::surfaces.back()->id_, i_periodic);
            int hi_id = std::max(model::surfaces.back()->id_, i_periodic);
            periodic_pairs.insert({lo_id, hi_id});
          } else {
            periodic_pairs.insert({model::surfaces.back()->id_, -1});
          }
        }
      }
    }
  }

  // Fill the surface map
  for (int i_surf = 0; i_surf < model::surfaces.size(); i_surf++) {
    int id = model::surfaces[i_surf]->id_;
    auto in_map = model::surface_map.find(id);
    if (in_map == model::surface_map.end()) {
      model::surface_map[id] = i_surf;
    } else {
      fatal_error(
        fmt::format("Two or more surfaces use the same unique ID: {}", id));
    }
  }
}

void prepare_boundary_conditions(std::set<std::pair<int, int>>& periodic_pairs,
  std::unordered_map<int, double>& albedo_map,
  std::unordered_map<int, int>& periodic_sense_map)
{
  // Fill the senses map for periodic surfaces
  auto n_periodic = periodic_sense_map.size();
  for (const auto& cell : model::cells) {
    if (n_periodic == 0)
      break; // Early exit once all periodic surfaces found

    for (auto s : cell->surfaces()) {
      auto surf_idx = std::abs(s) - 1;
      auto id = model::surfaces[surf_idx]->id_;

      if (periodic_sense_map.count(id)) {
        periodic_sense_map[id] = std::copysign(1, s);
        --n_periodic;
      }
    }
  }

  // Resolve unpaired periodic surfaces.  A lambda function is used with
  // std::find_if to identify the unpaired surfaces.
  auto is_unresolved_pair = [](const std::pair<int, int> p) {
    return p.second == -1;
  };
  auto first_unresolved = std::find_if(
    periodic_pairs.begin(), periodic_pairs.end(), is_unresolved_pair);
  if (first_unresolved != periodic_pairs.end()) {
    // Found one unpaired surface; search for a second one
    auto next_elem = first_unresolved;
    next_elem++;
    auto second_unresolved =
      std::find_if(next_elem, periodic_pairs.end(), is_unresolved_pair);
    if (second_unresolved == periodic_pairs.end()) {
      fatal_error("Found only one periodic surface without a specified partner."
                  " Please specify the partner for each periodic surface.");
    }

    // Make sure there isn't a third unpaired surface
    next_elem = second_unresolved;
    next_elem++;
    auto third_unresolved =
      std::find_if(next_elem, periodic_pairs.end(), is_unresolved_pair);
    if (third_unresolved != periodic_pairs.end()) {
      fatal_error(
        "Found at least three periodic surfaces without a specified "
        "partner. Please specify the partner for each periodic surface.");
    }

    // Add the completed pair and remove the old, unpaired entries
    int lo_id = std::min(first_unresolved->first, second_unresolved->first);
    int hi_id = std::max(first_unresolved->first, second_unresolved->first);
    periodic_pairs.insert({lo_id, hi_id});
    periodic_pairs.erase(first_unresolved);
    periodic_pairs.erase(second_unresolved);
  }

  // Assign the periodic boundary conditions with albedos
  for (auto periodic_pair : periodic_pairs) {
    int i_surf = model::surface_map[periodic_pair.first];
    int j_surf = model::surface_map[periodic_pair.second];
    Surface& surf1 {*model::surfaces[i_surf]};
    Surface& surf2 {*model::surfaces[j_surf]};

    // Compute the dot product of the surface normals
    Direction norm1 = surf1.normal({0, 0, 0});
    Direction norm2 = surf2.normal({0, 0, 0});
    norm1 /= norm1.norm();
    norm2 /= norm2.norm();
    double dot_prod = norm1.dot(norm2);

    // If the dot product is 1 (to within floating point precision) then the
    // planes are parallel which indicates a translational periodic boundary
    // condition.  Otherwise, it is a rotational periodic BC.
    if (std::abs(1.0 - dot_prod) < FP_PRECISION) {
      surf1.bc_ = make_unique<TranslationalPeriodicBC>(i_surf, j_surf);
      surf2.bc_ = make_unique<TranslationalPeriodicBC>(j_surf, i_surf);
    } else {
      // check that both normals have at least one 0 component
      if (std::abs(norm1.x) > FP_PRECISION &&
          std::abs(norm1.y) > FP_PRECISION &&
          std::abs(norm1.z) > FP_PRECISION) {
        fatal_error(fmt::format(
          "The normal ({}) of the periodic surface ({}) does not contain any "
          "component with a zero value. A RotationalPeriodicBC requires one "
          "component which is zero for both plane normals.",
          norm1, i_surf));
      }
      if (std::abs(norm2.x) > FP_PRECISION &&
          std::abs(norm2.y) > FP_PRECISION &&
          std::abs(norm2.z) > FP_PRECISION) {
        fatal_error(fmt::format(
          "The normal ({}) of the periodic surface ({}) does not contain any "
          "component with a zero value. A RotationalPeriodicBC requires one "
          "component which is zero for both plane normals.",
          norm2, j_surf));
      }
      // find common zero component, which indicates the periodic axis
      RotationalPeriodicBC::PeriodicAxis axis;
      if (std::abs(norm1.x) <= FP_PRECISION &&
          std::abs(norm2.x) <= FP_PRECISION) {
        axis = RotationalPeriodicBC::PeriodicAxis::x;
      } else if (std::abs(norm1.y) <= FP_PRECISION &&
                 std::abs(norm2.y) <= FP_PRECISION) {
        axis = RotationalPeriodicBC::PeriodicAxis::y;
      } else if (std::abs(norm1.z) <= FP_PRECISION &&
                 std::abs(norm2.z) <= FP_PRECISION) {
        axis = RotationalPeriodicBC::PeriodicAxis::z;
      } else {
        fatal_error(fmt::format(
          "There is no component which is 0.0 in both normal vectors. This "
          "indicates that the two planes are not periodic about the X, Y, or Z "
          "axis, which is not supported."));
      }
      auto i_sign = periodic_sense_map[periodic_pair.first];
      auto j_sign = periodic_sense_map[periodic_pair.second];
      surf1.bc_ = make_unique<RotationalPeriodicBC>(
        i_sign * (i_surf + 1), j_sign * (j_surf + 1), axis);
      surf2.bc_ = make_unique<RotationalPeriodicBC>(
        j_sign * (j_surf + 1), i_sign * (i_surf + 1), axis);
    }

    // If albedo data is present in albedo map, set the boundary albedo.
    if (albedo_map.count(surf1.id_)) {
      surf1.bc_->set_albedo(albedo_map[surf1.id_]);
    }
    if (albedo_map.count(surf2.id_)) {
      surf2.bc_->set_albedo(albedo_map[surf2.id_]);
    }
  }
}

void free_memory_surfaces()
{
  model::surfaces.clear();
  model::surface_map.clear();
}

//==============================================================================
// Shortest distance from a point to a surface
//
// Each of these is the exact nearest-point distance, not an approximation: a
// sphere of the returned radius about r provably does not touch the surface.
// The one exception is a quadric that admits no exact answer at all -- one
// with no real points, or a repeated plane, whose gradient vanishes along the
// surface -- which falls back to a bound that still never over-estimates.
//
// The cones reduce exactly: a surface of revolution seen in the half-plane of
// (axial offset from the apex, radial distance from the axis) is the pair of
// lines rho = +/- s*a, and folding on |a| leaves a single ray whose projection
// of the point is never negative, so the perpendicular distance always applies.
//==============================================================================

namespace {

//! Distance from a point to a cone, given its offset along the axis from the
//! apex, its distance from the axis, and the square of the cone's slope.
double cone_point_distance(double axial, double radial, double slope_sq)
{
  double slope = std::sqrt(slope_sq);
  return std::abs(slope * std::abs(axial) - radial) / std::sqrt(1.0 + slope_sq);
}

//! Distance from a point to an origin-centred, axis-aligned ellipse.
//!
//! The ellipse is p^2/ep^2 + q^2/eq^2 = 1. Reflecting into the first quadrant
//! and ordering the semi-axes, the closest point is (ep^2 p/(s + ep^2),
//! eq^2 q/(s + eq^2)) for the unique s > -eq^2 that puts it back on the
//! ellipse. Written in the normalised variables below that condition is
//! strictly decreasing in s, so the root is bracketed and bisected rather
//! than found from the quartic, which keeps it robust for every aspect ratio
//! and every position of the point.
double ellipse_point_distance(double p, double q, double ep, double eq)
{
  // A circular cross section needs no iteration
  if (ep == eq)
    return std::abs(std::sqrt(p * p + q * q) - ep);

  // Reflect into the first quadrant and put the major semi-axis first
  p = std::abs(p);
  q = std::abs(q);
  if (ep < eq) {
    std::swap(p, q);
    std::swap(ep, eq);
  }

  if (q == 0.0) {
    // On the major axis. Inside the evolute the nearest point leaves the
    // axis; outside it the vertex itself is nearest.
    double numer = ep * p;
    double denom = ep * ep - eq * eq;
    if (numer < denom) {
      double ratio = numer / denom;
      double x = ep * ratio;
      double y = eq * std::sqrt(std::max(0.0, 1.0 - ratio * ratio));
      return std::sqrt((x - p) * (x - p) + y * y);
    }
    return std::abs(p - ep);
  }

  if (p == 0.0) {
    // On the minor axis, where the vertex is always nearest
    return std::abs(q - eq);
  }

  double z0 = p / ep;
  double z1 = q / eq;
  double g = z0 * z0 + z1 * z1 - 1.0;
  if (g == 0.0)
    return 0.0;

  // Bracket the root of g(s) = (r0 z0/(s + r0))^2 + (z1/(s + 1))^2 - 1. An
  // interior point (g < 0) has its root at s <= 0, an exterior point above it.
  double r0 = (ep / eq) * (ep / eq);
  double n0 = r0 * z0;
  double s0 = z1 - 1.0;
  double s1 = (g < 0.0) ? 0.0 : std::sqrt(n0 * n0 + z1 * z1) - 1.0;

  double s = 0.5 * (s0 + s1);
  // Bisection on a double exhausts the mantissa in at most ~64 halvings; the
  // loop bound is a backstop and the s == s0 || s == s1 test is what ends it.
  for (int i = 0; i < 100; ++i) {
    s = 0.5 * (s0 + s1);
    if (s == s0 || s == s1)
      break;
    double ratio0 = n0 / (s + r0);
    double ratio1 = z1 / (s + 1.0);
    double gs = ratio0 * ratio0 + ratio1 * ratio1 - 1.0;
    if (gs > 0.0) {
      s0 = s;
    } else if (gs < 0.0) {
      s1 = s;
    } else {
      break;
    }
  }

  double x = r0 * p / (s + r0);
  double y = q / (s + 1.0);
  return std::sqrt((x - p) * (x - p) + (y - q) * (y - q));
}

//==============================================================================
// Exact nearest point on a general quadric
//
// Re-centre the quadric on the query point, so we look for the offset y that
// minimises |y| subject to y^T M y + g.y + q = 0, with M the matrix of
// second-order coefficients, g the gradient at the query point and q the
// quadric evaluated there. At the nearest point the surface normal is parallel
// to y, so for some multiplier lambda
//
//     (I + lambda M) y = -lambda g / 2
//
// Rotating into the eigenbasis of M decouples that into three scalar
// equations, (1 + lambda d_i) y_i = -lambda b_i / 2, and the decoupled form is
// what makes the degenerate cases visible rather than silently lost:
//
//   * where 1 + lambda d_i != 0 the component is determined, y_i =
//     -lambda b_i / (2 (1 + lambda d_i));
//   * where 1 + lambda d_i == 0 the equation reads 0 = -lambda b_i / 2, so it
//     admits a solution only when b_i == 0, and then leaves y_i FREE, pinned
//     only by the surface equation itself.
//
// That second case is not an edge case to be waved at. It is exactly what
// happens whenever the query point lies on a symmetry plane of the quadric --
// a point on the axis of an ellipsoid, say -- which is the common situation in
// a reactor model, and it is usually where the true minimum lives. Solving the
// system by Cramer's rule instead and clearing the denominator, as one would
// to get a single sextic in lambda, drops precisely these solutions: their
// multiplier is a pole of y = u/det. So the free directions are carried
// explicitly below.
//
// Substituting the determined components into the surface equation and
// multiplying by prod_i (1 + lambda d_i)^2 leaves a sextic whose real roots
// are the non-degenerate critical points. Completeness is the whole game
// here -- a missed root returns too large a distance and would wrongly certify
// a sphere as clear, which is the one failure this routine must not have -- so
// the candidate multipliers are every real root of that sextic, every real
// root of its derivative (a tangency shows up as a double root, which has no
// sign change to bracket), and every pole -1/d_i. Each candidate is then
// checked against the surface equation, so a spurious one is discarded rather
// than believed.
//==============================================================================

//! A polynomial in the Lagrange multiplier, ascending powers. Six is the
//! highest degree the construction above can produce.
struct LambdaPoly {
  static constexpr int MAX_DEGREE = 6;
  double c[MAX_DEGREE + 1] {};
};

LambdaPoly poly_add(const LambdaPoly& a, const LambdaPoly& b)
{
  LambdaPoly r;
  for (int i = 0; i <= LambdaPoly::MAX_DEGREE; ++i)
    r.c[i] = a.c[i] + b.c[i];
  return r;
}

LambdaPoly poly_mul(const LambdaPoly& a, const LambdaPoly& b)
{
  LambdaPoly r;
  for (int i = 0; i <= LambdaPoly::MAX_DEGREE; ++i) {
    if (a.c[i] == 0.0)
      continue;
    for (int j = 0; i + j <= LambdaPoly::MAX_DEGREE; ++j)
      r.c[i + j] += a.c[i] * b.c[j];
  }
  return r;
}

LambdaPoly poly_scale(double s, const LambdaPoly& a)
{
  LambdaPoly r;
  for (int i = 0; i <= LambdaPoly::MAX_DEGREE; ++i)
    r.c[i] = s * a.c[i];
  return r;
}

//! Horner evaluation of a polynomial given in ascending powers
double poly_eval(const double* c, int n, double x)
{
  double v = c[n];
  for (int i = n - 1; i >= 0; --i)
    v = v * x + c[i];
  return v;
}

//! Refine a root already bracketed by a sign change over [lo, hi].
double poly_bisect(const double* c, int n, double lo, double hi)
{
  double flo = poly_eval(c, n, lo);
  while (true) {
    double mid = 0.5 * (lo + hi);
    if (mid <= lo || mid >= hi)
      return mid;
    double fmid = poly_eval(c, n, mid);
    if (fmid == 0.0)
      return mid;
    if ((fmid > 0.0) == (flo > 0.0)) {
      lo = mid;
      flo = fmid;
    } else {
      hi = mid;
    }
  }
}

//! Every real root of a polynomial, written to `out` in ascending order.
//!
//! Complete for simple roots by construction, which is the property the caller
//! depends on: the real roots of the derivative split the line into intervals
//! on which the polynomial is monotone, so each interval holds at most one
//! root and a sign change at its ends brackets it for bisection. Recursing on
//! the derivative bottoms out at a linear polynomial, and Cauchy's bound
//! closes off the two outer intervals. A root of even multiplicity has no sign
//! change and is not found here; the caller covers that by also trying the
//! roots of the derivative, which is where such a root also lies.
//! \return the number of roots written
int poly_real_roots(const double* c, int n, double* out)
{
  // Work with the true degree. A vanishing leading coefficient is routine
  // here, since the sextic degenerates whenever M is singular.
  while (n > 0 && c[n] == 0.0)
    --n;
  if (n < 1)
    return 0;
  if (n == 1) {
    out[0] = -c[0] / c[1];
    return 1;
  }

  double deriv[LambdaPoly::MAX_DEGREE + 1];
  for (int i = 1; i <= n; ++i)
    deriv[i - 1] = i * c[i];
  double crit[LambdaPoly::MAX_DEGREE + 1];
  int n_crit = poly_real_roots(deriv, n - 1, crit);

  // Cauchy's bound: every real root is smaller than this in modulus
  double largest = 0.0;
  for (int i = 0; i < n; ++i)
    largest = std::max(largest, std::abs(c[i]));
  double limit = 1.0 + largest / std::abs(c[n]);

  double ends[LambdaPoly::MAX_DEGREE + 3];
  int n_ends = 0;
  ends[n_ends++] = -limit;
  for (int i = 0; i < n_crit; ++i) {
    if (crit[i] > ends[n_ends - 1] && crit[i] < limit)
      ends[n_ends++] = crit[i];
  }
  ends[n_ends++] = limit;

  int n_roots = 0;
  for (int i = 0; i + 1 < n_ends; ++i) {
    double flo = poly_eval(c, n, ends[i]);
    double fhi = poly_eval(c, n, ends[i + 1]);
    if (flo == 0.0) {
      out[n_roots++] = ends[i];
    } else if (fhi == 0.0) {
      out[n_roots++] = ends[i + 1];
    } else if ((flo > 0.0) != (fhi > 0.0)) {
      out[n_roots++] = poly_bisect(c, n, ends[i], ends[i + 1]);
    }
  }
  return n_roots;
}

//! Eigenvalues and orthonormal eigenvectors of a symmetric 3x3 matrix, by
//! cyclic Jacobi rotations. Preferred over the closed form because it stays
//! accurate for repeated and near-repeated eigenvalues, which are the rule
//! rather than the exception here: every sphere, cylinder, cone and surface
//! of revolution has them.
//! \param[out] value the three eigenvalues
//! \param[out] basis columns are the corresponding unit eigenvectors
void symmetric_eigen(const double in[3][3], double value[3], double basis[3][3])
{
  double a[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      a[i][j] = in[i][j];
      basis[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }

  for (int sweep = 0; sweep < 50; ++sweep) {
    if (std::abs(a[0][1]) + std::abs(a[0][2]) + std::abs(a[1][2]) == 0.0)
      break;
    for (int p = 0; p < 2; ++p) {
      for (int r = p + 1; r < 3; ++r) {
        if (a[p][r] == 0.0)
          continue;
        double theta = 0.5 * (a[r][r] - a[p][p]) / a[p][r];
        double t = (theta >= 0.0 ? 1.0 : -1.0) /
                   (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        double c = 1.0 / std::sqrt(t * t + 1.0);
        double s = t * c;
        for (int k = 0; k < 3; ++k) {
          double akp = a[k][p], akr = a[k][r];
          a[k][p] = c * akp - s * akr;
          a[k][r] = s * akp + c * akr;
        }
        for (int k = 0; k < 3; ++k) {
          double apk = a[p][k], ark = a[r][k];
          a[p][k] = c * apk - s * ark;
          a[r][k] = s * apk + c * ark;
        }
        for (int k = 0; k < 3; ++k) {
          double vkp = basis[k][p], vkr = basis[k][r];
          basis[k][p] = c * vkp - s * vkr;
          basis[k][r] = s * vkp + c * vkr;
        }
      }
    }
  }
  for (int i = 0; i < 3; ++i)
    value[i] = a[i][i];
}

// Everything below works on a quadric normalised so that no coefficient
// exceeds one in magnitude, which lets these be plain absolute tolerances.
// Eigenvalues closer than this are treated as one repeated eigenvalue.
constexpr double QUADRIC_EIGEN_TOL = 1e-10;
// Below this, a gradient component is taken to be absent, which is what lets
// an eigenspace go free. Larger components are handled by the ordinary roots.
constexpr double QUADRIC_POLE_TOL = 1e-10;
// How far the surface equation may miss zero before a candidate point is
// judged not to lie on the surface at all. Loose next to the precision a
// polished multiplier actually reaches, because its only job is to separate
// genuine roots from the spurious candidates tried alongside them, and those
// miss by many orders of magnitude rather than by a few digits.
constexpr double QUADRIC_SURFACE_TOL = 1e-6;

//! A Lagrange multiplier, held as an offset from one eigenspace's pole.
//!
//! The multipliers that matter cluster around the poles -1/d_G, and the
//! denominator 1 + lambda d_G is then a difference of two nearly equal
//! numbers: forming it directly throws away exactly the digits the answer
//! depends on, which shows up as a distance that is slightly too large --
//! precisely the direction distance_to_point promises never to err in. Naming
//! the pole and carrying the offset instead lets that denominator be built as
//! d_G * offset, with no cancellation at all. Newton then converges on the
//! offset, so the lost digits are not merely avoided but recovered: the fixed
//! point is set by the equation, not by the starting value.
struct Multiplier {
  int pole {-1};       //!< eigenspace whose denominator vanishes, or -1
  double offset {0.0}; //!< the multiplier itself when pole is -1
};

double multiplier_value(const Multiplier& mu, const double dg[3])
{
  return mu.pole < 0 ? mu.offset : mu.offset - 1.0 / dg[mu.pole];
}

//! The denominator 1 + lambda d_i, built without cancellation at the pole
double multiplier_denominator(const Multiplier& mu, int i, const double dg[3])
{
  if (i == mu.pole)
    return dg[i] * mu.offset;
  return 1.0 + multiplier_value(mu, dg) * dg[i];
}

//! The surface equation as a function of the multiplier, and its derivative
//! with respect to the offset.
//!
//! With the determined components substituted the surface equation becomes
//!
//!     F = q - sum_G b_G^2 (lambda/4)(d_G lambda + 2) / (1 + lambda d_G)^2
//!
//! whose derivative collapses, once the numerator algebra cancels, to
//!
//!     F' = -sum_G b_G^2 / (2 (1 + lambda d_G)^3)
//!
//! and the offset differs from lambda only by a constant, so the two
//! derivatives coincide.
void quadric_equation(const Multiplier& mu, const double dg[3],
  const double bsq[3], int n_group, double q, double& f, double& df)
{
  double lambda = multiplier_value(mu, dg);
  f = q;
  df = 0.0;
  for (int i = 0; i < n_group; ++i) {
    double s = multiplier_denominator(mu, i, dg);
    double s2 = s * s;
    f -= bsq[i] * (0.25 * lambda) * (dg[i] * lambda + 2.0) / s2;
    df -= 0.5 * bsq[i] / (s2 * s);
  }
}

//! Newton refinement of a multiplier on the surface equation itself.
//!
//! The polynomial is formed by clearing the denominators, which near a pole
//! multiplies the equation by something vanishingly small and leaves its root
//! poorly resolved -- and the roots that matter are the ones near poles. The
//! surface equation has no such problem: it is steep exactly where the
//! polynomial is flat. Steps are clamped to the pole-free interval the
//! starting point lies in, so the iteration cannot cross to another root.
Multiplier quadric_polish(
  Multiplier mu, const double dg[3], const double bsq[3], int n_group, double q)
{
  // The offsets at which a neighbouring eigenspace's denominator vanishes
  double base = multiplier_value(mu, dg) - mu.offset;
  double lo = -INFTY;
  double hi = INFTY;
  for (int i = 0; i < n_group; ++i) {
    if (dg[i] == 0.0)
      continue;
    double edge = -1.0 / dg[i] - base;
    if (edge < mu.offset)
      lo = std::max(lo, edge);
    else if (edge > mu.offset)
      hi = std::min(hi, edge);
  }

  double f, df;
  quadric_equation(mu, dg, bsq, n_group, q, f, df);
  Multiplier best = mu;
  double best_f = std::abs(f);
  for (int it = 0; it < 60; ++it) {
    if (df == 0.0 || !std::isfinite(f) || !std::isfinite(df))
      break;
    double next = mu.offset - f / df;
    if (!(next > lo && next < hi) || next == mu.offset || !std::isfinite(next))
      break;
    mu.offset = next;
    quadric_equation(mu, dg, bsq, n_group, q, f, df);
    if (std::abs(f) < best_f) {
      best_f = std::abs(f);
      best = mu;
    }
  }
  return best;
}

//! Squared distance to the surface point produced by one candidate multiplier.
//!
//! \param mu the trial multiplier
//! \param dg the distinct eigenvalues of the (normalised) second-order matrix
//! \param bsq for each, the squared length of the (normalised) gradient's
//!   component in that eigenspace -- all the geometry that survives, since
//!   nothing here depends on the individual directions within an eigenspace
//! \param n_group how many distinct eigenvalues there are
//! \param q the (normalised) quadric evaluated at the query point
//! \return the squared distance, or a negative value if this multiplier does
//!   not correspond to a point of the surface
double quadric_candidate(
  Multiplier mu, const double dg[3], const double bsq[3], int n_group, double q)
{
  // A multiplier sitting exactly on a pole is the free-eigenspace case, which
  // the surface equation cannot refine; anything else is polished first
  bool on_pole = mu.pole >= 0 && mu.offset == 0.0;
  if (!on_pole)
    mu = quadric_polish(mu, dg, bsq, n_group, q);

  double lambda = multiplier_value(mu, dg);
  double fixed_sq = 0.0;
  double rest = q;
  double rest_scale = std::abs(q);
  int n_free = 0;
  double d_free = 0.0;

  for (int i = 0; i < n_group; ++i) {
    double s = multiplier_denominator(mu, i, dg);
    if (s == 0.0) {
      // This eigenspace's equation has collapsed to 0 = -lambda b_G / 2. It
      // is satisfiable only if the gradient has no component here, and then
      // the component of y in this eigenspace is free, pinned only by the
      // surface equation itself.
      if (bsq[i] > QUADRIC_POLE_TOL * QUADRIC_POLE_TOL)
        return -1.0;
      d_free = dg[i];
      ++n_free;
      continue;
    }
    double factor = -0.5 * lambda / s;
    double y_sq = factor * factor * bsq[i];
    fixed_sq += y_sq;
    rest += dg[i] * y_sq + factor * bsq[i];
    rest_scale = std::max(
      rest_scale, std::max(std::abs(dg[i] * y_sq), std::abs(factor * bsq[i])));
  }

  if (n_free == 0) {
    // `rest` is the surface equation evaluated at y, which a genuine critical
    // point drives to zero
    if (std::abs(rest) > QUADRIC_SURFACE_TOL * std::max(1.0, rest_scale))
      return -1.0;
    return fixed_sq;
  }

  if (d_free == 0.0)
    return -1.0;
  double free_sq = -rest / d_free;
  if (free_sq < 0.0)
    return -1.0;
  return fixed_sq + free_sq;
}

//! Turn a bare multiplier into one held against whichever pole it is nearest,
//! so that the denominator it makes small is never formed by cancellation.
Multiplier quadric_anchor(double lambda, const double dg[3], int n_group)
{
  Multiplier mu;
  mu.offset = lambda;
  double closest = INFTY;
  for (int i = 0; i < n_group; ++i) {
    if (dg[i] == 0.0)
      continue;
    double s = std::abs(1.0 + lambda * dg[i]);
    if (s < 0.5 && s < closest) {
      closest = s;
      mu.pole = i;
      mu.offset = lambda + 1.0 / dg[i];
    }
  }
  return mu;
}

//! A rigorous lower bound on the distance from a point to a quadric.
//!
//! Used where the exact solve has nothing to return: a quadric with no real
//! points, or a degenerate one whose gradient vanishes along the surface
//! itself -- a repeated plane, say -- so that no Lagrange multiplier exists
//! for any surface point. Since f is quadratic, a step h from r satisfies
//!
//!     |f(r + h) - f(r)| <= |grad f(r)| |h| + ||M|| |h|^2
//!
//! so no zero of f lies within a distance d of r while the right-hand side
//! stays below |f(r)|. The largest such d is the positive root of
//! ||M|| d^2 + |grad f(r)| d - |f(r)| = 0. Over-estimating ||M|| keeps the
//! bound rigorous and merely loosens it, so the largest absolute row sum
//! stands in for the spectral norm; the two agree whenever M is diagonal.
double quadric_lower_bound(const double m[3][3], const double g[3], double q)
{
  double f = std::abs(q);
  if (f == 0.0)
    return 0.0;
  double grad = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
  double norm = 0.0;
  for (int i = 0; i < 3; ++i) {
    double row = 0.0;
    for (int j = 0; j < 3; ++j)
      row += std::abs(m[i][j]);
    norm = std::max(norm, row);
  }
  if (norm == 0.0)
    return grad > 0.0 ? f / grad : -1.0;
  return (std::sqrt(grad * grad + 4.0 * norm * f) - grad) / (2.0 * norm);
}

//! Distance from a point to a torus, given the point's offset along the axis
//! of revolution from the torus centre and its distance from that axis.
//!
//! A torus is a surface of revolution, so the nearest surface point lies in
//! the half-plane spanned by the query point and the axis: for a surface
//! point at azimuth phi out of that half-plane the squared distance depends
//! on phi only through a -2 * radial * rho * cos(phi) term, and radial and
//! rho are both non-negative, so phi = 0 always minimises it. What remains is
//! the distance in that half-plane to the generating ellipse, which is
//! centred at (axial 0, radial A) with semi-axis B along the axis and C
//! across it.
//!
//! For a proper torus, A > C, the whole generating ellipse sits at positive
//! radius and the result is exact. A degenerate A <= C puts part of the
//! ellipse at negative radius, where it is not a point of the surface;
//! minimising over those as well can only return something smaller, so the
//! result remains a valid lower bound.
double torus_point_distance(
  double axial, double radial, double A, double B, double C)
{
  return ellipse_point_distance(axial, radial - A, B, C);
}

} // namespace

double SurfaceXPlane::distance_to_point(Position r) const
{
  return std::abs(r.x - x0_);
}

double SurfaceYPlane::distance_to_point(Position r) const
{
  return std::abs(r.y - y0_);
}

double SurfaceZPlane::distance_to_point(Position r) const
{
  return std::abs(r.z - z0_);
}

double SurfacePlane::distance_to_point(Position r) const
{
  return std::abs(A_ * r.x + B_ * r.y + C_ * r.z - D_) /
         std::sqrt(A_ * A_ + B_ * B_ + C_ * C_);
}

double SurfaceXCylinder::distance_to_point(Position r) const
{
  double y = r.y - y0_;
  double z = r.z - z0_;
  return std::abs(std::sqrt(y * y + z * z) - radius_);
}

double SurfaceYCylinder::distance_to_point(Position r) const
{
  double x = r.x - x0_;
  double z = r.z - z0_;
  return std::abs(std::sqrt(x * x + z * z) - radius_);
}

double SurfaceZCylinder::distance_to_point(Position r) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  return std::abs(std::sqrt(x * x + y * y) - radius_);
}

double SurfaceSphere::distance_to_point(Position r) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  double z = r.z - z0_;
  return std::abs(std::sqrt(x * x + y * y + z * z) - radius_);
}

double SurfaceXCone::distance_to_point(Position r) const
{
  double y = r.y - y0_;
  double z = r.z - z0_;
  return cone_point_distance(r.x - x0_, std::sqrt(y * y + z * z), radius_sq_);
}

double SurfaceYCone::distance_to_point(Position r) const
{
  double x = r.x - x0_;
  double z = r.z - z0_;
  return cone_point_distance(r.y - y0_, std::sqrt(x * x + z * z), radius_sq_);
}

double SurfaceZCone::distance_to_point(Position r) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  return cone_point_distance(r.z - z0_, std::sqrt(x * x + y * y), radius_sq_);
}

double SurfaceQuadric::distance_to_point(Position r) const
{
  // Re-express the quadric about r, so we solve for the offset y = x - r:
  //     y^T m y + g.y + q = 0
  // Centring on the query point is what keeps the coefficients in a sane
  // range: they then describe the local geometry rather than the surface's
  // distance from the global origin.
  double m[3][3] = {{A_, 0.5 * D_, 0.5 * F_}, {0.5 * D_, B_, 0.5 * E_},
    {0.5 * F_, 0.5 * E_, C_}};
  double g[3] = {2.0 * A_ * r.x + D_ * r.y + F_ * r.z + G_,
    2.0 * B_ * r.y + D_ * r.x + E_ * r.z + H_,
    2.0 * C_ * r.z + E_ * r.y + F_ * r.x + J_};
  double q = evaluate(r);

  if (q == 0.0)
    return 0.0;

  // Dividing the equation through by a constant changes neither the surface
  // nor the answer, only the scale of the multiplier
  double scale = std::abs(q);
  for (int i = 0; i < 3; ++i) {
    scale = std::max(scale, std::abs(g[i]));
    for (int j = 0; j < 3; ++j)
      scale = std::max(scale, std::abs(m[i][j]));
  }
  if (scale == 0.0)
    return -1.0;
  for (int i = 0; i < 3; ++i) {
    g[i] /= scale;
    for (int j = 0; j < 3; ++j)
      m[i][j] /= scale;
  }
  q /= scale;

  double d[3];
  double basis[3][3];
  symmetric_eigen(m, d, basis);

  // The gradient in the eigenbasis
  double b[3];
  for (int i = 0; i < 3; ++i)
    b[i] = basis[0][i] * g[0] + basis[1][i] * g[1] + basis[2][i] * g[2];

  // Collect repeated eigenvalues into one eigenspace each. Only the
  // eigenvalue and the squared length of the gradient's component in the
  // eigenspace ever matter, never the individual directions within it, and
  // grouping is what keeps the common surfaces conditioned: a sphere, a
  // cylinder, a cone or any surface of revolution has a repeated eigenvalue,
  // which left ungrouped would raise the polynomial's degree and stack
  // several of its roots on the same point, where they cannot be resolved.
  double dg[3];
  double bsq[3] {};
  int n_group = 0;
  double d_scale =
    std::max({std::abs(d[0]), std::abs(d[1]), std::abs(d[2]), 1.0});
  for (int i = 0; i < 3; ++i) {
    int group = -1;
    for (int k = 0; k < n_group; ++k)
      if (std::abs(d[i] - dg[k]) <= QUADRIC_EIGEN_TOL * d_scale)
        group = k;
    if (group < 0) {
      dg[n_group] = d[i];
      group = n_group++;
    }
    bsq[group] += b[i] * b[i];
  }

  // The surface equation with the determined components substituted,
  // multiplied through by prod_G (1 + lambda d_G)^2:
  //     q prod_G (1 + lambda d_G)^2
  //       - sum_G b_G^2 (lambda/4)(d_G lambda + 2) prod_{H != G} (...)^2
  LambdaPoly squared[3];
  for (int i = 0; i < n_group; ++i) {
    LambdaPoly linear;
    linear.c[0] = 1.0;
    linear.c[1] = dg[i];
    squared[i] = poly_mul(linear, linear);
  }

  LambdaPoly all;
  all.c[0] = 1.0;
  for (int i = 0; i < n_group; ++i)
    all = poly_mul(all, squared[i]);

  LambdaPoly poly = poly_scale(q, all);
  for (int i = 0; i < n_group; ++i) {
    if (bsq[i] == 0.0)
      continue;
    LambdaPoly others;
    others.c[0] = 1.0;
    for (int j = 0; j < n_group; ++j)
      if (j != i)
        others = poly_mul(others, squared[j]);
    LambdaPoly factor;
    factor.c[1] = -0.5;
    factor.c[2] = -0.25 * dg[i];
    poly = poly_add(poly, poly_scale(bsq[i], poly_mul(factor, others)));
  }

  // Candidate multipliers: every real root of that polynomial, every real
  // root of its derivative so that a tangency appearing as a root of even
  // multiplicity is not passed over, and every pole, which is where an
  // eigenspace goes free. Each is checked against the surface equation
  // afterwards, so offering too many costs nothing but offering too few would
  // return a distance that is too large.
  double roots[2 * LambdaPoly::MAX_DEGREE + 4];
  int n = poly_real_roots(poly.c, LambdaPoly::MAX_DEGREE, roots);

  double deriv[LambdaPoly::MAX_DEGREE + 1];
  for (int i = 1; i <= LambdaPoly::MAX_DEGREE; ++i)
    deriv[i - 1] = i * poly.c[i];
  n += poly_real_roots(deriv, LambdaPoly::MAX_DEGREE - 1, roots + n);

  Multiplier candidates[3 * LambdaPoly::MAX_DEGREE + 8];
  int n_cand = 0;
  for (int k = 0; k < n; ++k)
    candidates[n_cand++] = quadric_anchor(roots[k], dg, n_group);
  for (int i = 0; i < n_group; ++i) {
    if (dg[i] == 0.0)
      continue;
    Multiplier at_pole;
    at_pole.pole = i;
    at_pole.offset = 0.0;
    candidates[n_cand++] = at_pole;
  }

  double best_sq = INFTY;
  for (int k = 0; k < n_cand; ++k) {
    double d_sq = quadric_candidate(candidates[k], dg, bsq, n_group, q);
    if (d_sq >= 0.0 && std::isfinite(d_sq))
      best_sq = std::min(best_sq, d_sq);
  }

  if (best_sq < INFTY)
    return std::sqrt(best_sq);

  // Nothing satisfied the surface equation. Either the quadric has no real
  // points, or it is degenerate in a way that admits no multiplier anywhere
  // on the surface -- a repeated plane, whose gradient vanishes along the
  // surface itself. Neither leaves an exact distance to report, so fall back
  // to a bound that is still guaranteed not to over-estimate.
  return quadric_lower_bound(m, g, q);
}

double SurfaceXTorus::distance_to_point(Position r) const
{
  double y = r.y - y0_;
  double z = r.z - z0_;
  return torus_point_distance(r.x - x0_, std::sqrt(y * y + z * z), A_, B_, C_);
}

double SurfaceYTorus::distance_to_point(Position r) const
{
  double x = r.x - x0_;
  double z = r.z - z0_;
  return torus_point_distance(r.y - y0_, std::sqrt(x * x + z * z), A_, B_, C_);
}

double SurfaceZTorus::distance_to_point(Position r) const
{
  double x = r.x - x0_;
  double y = r.y - y0_;
  return torus_point_distance(r.z - z0_, std::sqrt(x * x + y * y), A_, B_, C_);
}

} // namespace openmc
