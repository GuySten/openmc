#include "openmc/volume_octree.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/lattice.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/mgxs_interface.h"
#include "openmc/nuclide.h"
#include "openmc/settings.h"
#include "openmc/surface.h"
#include "openmc/universe.h"

namespace openmc {

// volume_octree_math.h redefines the region operator tokens so it stays free of
// OpenMC dependencies and can be unit-tested standalone. Keep them honest.
static_assert(kOpLeftParen == OP_LEFT_PAREN, "operator token drift");
static_assert(kOpRightParen == OP_RIGHT_PAREN, "operator token drift");
static_assert(kOpComplement == OP_COMPLEMENT, "operator token drift");
static_assert(kOpIntersection == OP_INTERSECTION, "operator token drift");
static_assert(kOpUnion == OP_UNION, "operator token drift");

//==============================================================================
// Surface classification
//==============================================================================

bool surface_quadric_form(const Surface& surf, QuadricForm& q)
{
  q = QuadricForm {};

  if (auto s = dynamic_cast<const SurfaceXPlane*>(&surf)) {
    q.b = {1., 0., 0.};
    q.c = -s->x0_;
    return true;
  }
  if (auto s = dynamic_cast<const SurfaceYPlane*>(&surf)) {
    q.b = {0., 1., 0.};
    q.c = -s->y0_;
    return true;
  }
  if (auto s = dynamic_cast<const SurfaceZPlane*>(&surf)) {
    q.b = {0., 0., 1.};
    q.c = -s->z0_;
    return true;
  }
  if (auto s = dynamic_cast<const SurfacePlane*>(&surf)) {
    // A*x + B*y + C*z - D
    q.b = {s->A_, s->B_, s->C_};
    q.c = -s->D_;
    return true;
  }

  // (r1 - o1)^2 + (r2 - o2)^2 - radius^2
  if (auto s = dynamic_cast<const SurfaceXCylinder*>(&surf)) {
    q.A[1] = q.A[2] = 1.0;
    q.b = {0., -2. * s->y0_, -2. * s->z0_};
    q.c = s->y0_ * s->y0_ + s->z0_ * s->z0_ - s->radius_ * s->radius_;
    return true;
  }
  if (auto s = dynamic_cast<const SurfaceYCylinder*>(&surf)) {
    q.A[0] = q.A[2] = 1.0;
    q.b = {-2. * s->x0_, 0., -2. * s->z0_};
    q.c = s->x0_ * s->x0_ + s->z0_ * s->z0_ - s->radius_ * s->radius_;
    return true;
  }
  if (auto s = dynamic_cast<const SurfaceZCylinder*>(&surf)) {
    q.A[0] = q.A[1] = 1.0;
    q.b = {-2. * s->x0_, -2. * s->y0_, 0.};
    q.c = s->x0_ * s->x0_ + s->y0_ * s->y0_ - s->radius_ * s->radius_;
    return true;
  }

  if (auto s = dynamic_cast<const SurfaceSphere*>(&surf)) {
    q.A[0] = q.A[1] = q.A[2] = 1.0;
    q.b = {-2. * s->x0_, -2. * s->y0_, -2. * s->z0_};
    q.c = s->x0_ * s->x0_ + s->y0_ * s->y0_ + s->z0_ * s->z0_ -
          s->radius_ * s->radius_;
    return true;
  }

  // axis_aligned_cone_evaluate<i1,i2,i3>: r2^2 + r3^2 - radius_sq*r1^2,
  // where i1 is the cone axis.
  if (auto s = dynamic_cast<const SurfaceXCone*>(&surf)) {
    q.A[0] = -s->radius_sq_;
    q.A[1] = q.A[2] = 1.0;
    q.b = {2. * s->radius_sq_ * s->x0_, -2. * s->y0_, -2. * s->z0_};
    q.c = s->y0_ * s->y0_ + s->z0_ * s->z0_ -
          s->radius_sq_ * s->x0_ * s->x0_;
    return true;
  }
  if (auto s = dynamic_cast<const SurfaceYCone*>(&surf)) {
    q.A[1] = -s->radius_sq_;
    q.A[0] = q.A[2] = 1.0;
    q.b = {-2. * s->x0_, 2. * s->radius_sq_ * s->y0_, -2. * s->z0_};
    q.c = s->x0_ * s->x0_ + s->z0_ * s->z0_ -
          s->radius_sq_ * s->y0_ * s->y0_;
    return true;
  }
  if (auto s = dynamic_cast<const SurfaceZCone*>(&surf)) {
    q.A[2] = -s->radius_sq_;
    q.A[0] = q.A[1] = 1.0;
    q.b = {-2. * s->x0_, -2. * s->y0_, 2. * s->radius_sq_ * s->z0_};
    q.c = s->x0_ * s->x0_ + s->y0_ * s->y0_ -
          s->radius_sq_ * s->z0_ * s->z0_;
    return true;
  }

  // SurfaceQuadric::evaluate is
  //   x*(A*x + D*y + G) + y*(B*y + E*z + H) + z*(C*z + F*x + J) + K
  // so D is the xy term, E the yz term and F the xz term.
  if (auto s = dynamic_cast<const SurfaceQuadric*>(&surf)) {
    q.A[0] = s->A_;
    q.A[1] = s->B_;
    q.A[2] = s->C_;
    q.A[3] = 0.5 * s->D_; // xy
    q.A[4] = 0.5 * s->F_; // xz
    q.A[5] = 0.5 * s->E_; // yz
    q.b = {s->G_, s->H_, s->J_};
    q.c = s->K_;
    return true;
  }

  return false;
}

BoxSense classify_box(const Surface& surf, const OctBox& box)
{
  QuadricForm q;
  if (surface_quadric_form(surf, q)) {
    double lo, hi;
    form_range(q, box, lo, hi);
    return from_range(lo, hi);
  }

  // Tori are quartic, but separable into an axial and a radial term against an
  // unrotated box, so they get an exact bound of their own.
  if (!box.rotated) {
    int ax = -1;
    Position tc;
    double A = 0., B = 0., C = 0.;
    if (auto s = dynamic_cast<const SurfaceXTorus*>(&surf)) {
      ax = 0; tc = {s->x0_, s->y0_, s->z0_}; A = s->A_; B = s->B_; C = s->C_;
    } else if (auto s = dynamic_cast<const SurfaceYTorus*>(&surf)) {
      ax = 1; tc = {s->x0_, s->y0_, s->z0_}; A = s->A_; B = s->B_; C = s->C_;
    } else if (auto s = dynamic_cast<const SurfaceZTorus*>(&surf)) {
      ax = 2; tc = {s->x0_, s->y0_, s->z0_}; A = s->A_; B = s->B_; C = s->C_;
    }
    if (ax >= 0) {
      double lo, hi;
      torus_range(ax, tc, A, B, C, box, lo, hi);
      return from_range(lo, hi);
    }
  }

  // Rotated tori, DAGMC surfaces, anything unrecognised. BOTH is always sound:
  // it forces refinement and eventually charges slack.
  return BoxSense::BOTH;
}

//==============================================================================
// Setup
//==============================================================================

OctreeVolumeCalculation::OctreeVolumeCalculation(DomainType type,
  const vector<int>& domain_ids, Position lower_left, Position upper_right)
  : type_(type), domain_ids_(domain_ids)
{
  root_ = OctBox::from_aabb(lower_left, upper_right);
  results_.resize(domain_ids_.size());

  for (int i = 0; i < static_cast<int>(domain_ids_.size()); ++i) {
    int32_t id = domain_ids_[i];
    switch (type_) {
    case DomainType::CELL:
      cell_slot_[model::cell_map.at(id)] = i;
      break;
    case DomainType::MATERIAL:
      mat_slot_[model::material_map.at(id)] = i;
      break;
    case DomainType::UNIVERSE:
      univ_slot_[model::universe_map.at(id)] = i;
      break;
    }
  }

  // Pre-digest every surface. dynamic_cast in the classification hot loop was
  // costing more than the arithmetic it guarded.
  proxies_.resize(model::surfaces.size());
  for (size_t i = 0; i < model::surfaces.size(); ++i) {
    const Surface& s = *model::surfaces[i];
    auto& pr = proxies_[i];
    if (surface_quadric_form(s, pr.q)) {
      pr.kind = SurfaceProxy::Kind::QUADRIC;
      continue;
    }
    if (auto t = dynamic_cast<const SurfaceXTorus*>(&s)) {
      pr.kind = SurfaceProxy::Kind::TORUS;
      pr.ax = 0; pr.tc = {t->x0_, t->y0_, t->z0_};
      pr.A = t->A_; pr.B = t->B_; pr.C = t->C_;
    } else if (auto t = dynamic_cast<const SurfaceYTorus*>(&s)) {
      pr.kind = SurfaceProxy::Kind::TORUS;
      pr.ax = 1; pr.tc = {t->x0_, t->y0_, t->z0_};
      pr.A = t->A_; pr.B = t->B_; pr.C = t->C_;
    } else if (auto t = dynamic_cast<const SurfaceZTorus*>(&s)) {
      pr.kind = SurfaceProxy::Kind::TORUS;
      pr.ax = 2; pr.tc = {t->x0_, t->y0_, t->z0_};
      pr.A = t->A_; pr.B = t->B_; pr.C = t->C_;
    }
  }

  // Region expressions are stored in INFIX form; generate_postfix() rebuilds
  // the RPN with a shunting-yard pass on every call, so do it once here.
  size_t n = model::cells.size();
  postfix_.resize(n);
  simple_.assign(n, true);
  is_csg_.assign(n, false);
  for (size_t i = 0; i < n; ++i) {
    const auto* csg = dynamic_cast<const CSGCell*>(model::cells[i].get());
    if (!csg)
      continue; // DAGMC cell: left as non-CSG, always straddles
    is_csg_[i] = true;
    simple_[i] = csg->region().is_simple();
    postfix_[i] = simple_[i] ? csg->region().expression()
                             : csg->region().postfix(csg->id_);
  }

  // Enumerate what a slack charge beneath each universe has to touch. This
  // also registers every (domain, material) pair, which is what makes the
  // accumulators dense.
  reach_.resize(model::universes.size());
  reach(model::root_universe);

  lower_.assign(domain_ids_.size(), 0.0);
  slack_.assign(domain_ids_.size(), 0.0);
  mat_lower_.assign(pair_mat_.size(), 0.0);
  mat_slack_.assign(pair_mat_.size(), 0.0);
}

int OctreeVolumeCalculation::register_pair(int slot, int32_t i_mat)
{
  int64_t key = (static_cast<int64_t>(slot) << 32) | static_cast<uint32_t>(i_mat);
  auto it = pair_index_.find(key);
  if (it != pair_index_.end())
    return it->second;
  int idx = static_cast<int>(pair_mat_.size());
  pair_index_.emplace(key, idx);
  pair_mat_.push_back(i_mat);
  pair_slot_.push_back(slot);
  return idx;
}

int OctreeVolumeCalculation::find_pair(int slot, int32_t i_mat) const
{
  int64_t key = (static_cast<int64_t>(slot) << 32) | static_cast<uint32_t>(i_mat);
  auto it = pair_index_.find(key);
  return it == pair_index_.end() ? -1 : it->second;
}

const OctreeVolumeCalculation::Reach& OctreeVolumeCalculation::reach(
  int32_t i_univ)
{
  Reach& r = reach_[i_univ];
  if (r.done)
    return r;
  r.done = true; // also guards against a malformed cyclic geometry

  auto add_mats = [&](int slot, const Cell& c) {
    for (int32_t i_mat : c.material_)
      r.pairs.push_back(register_pair(slot, i_mat));
  };

  int u_slot = -1;
  auto iu = univ_slot_.find(i_univ);
  if (iu != univ_slot_.end()) {
    u_slot = iu->second;
    r.slots.push_back(u_slot);
  }

  const Universe& u = *model::universes[i_univ];
  for (int32_t i_cell : u.cells_) {
    const Cell& c = *model::cells[i_cell];

    int c_slot = -1;
    auto ic = cell_slot_.find(i_cell);
    if (ic != cell_slot_.end()) {
      c_slot = ic->second;
      r.slots.push_back(c_slot);
    }

    if (c.type_ == Fill::MATERIAL) {
      if (c_slot >= 0)
        add_mats(c_slot, c);
      if (u_slot >= 0)
        add_mats(u_slot, c);
      for (int32_t i_mat : c.material_) {
        auto im = mat_slot_.find(i_mat);
        if (im != mat_slot_.end()) {
          r.slots.push_back(im->second);
          r.pairs.push_back(register_pair(im->second, i_mat));
        }
      }
      continue;
    }

    // Collect the children, deduplicated. A lattice's universes_ lists the
    // same universe once per position; walking it naively is what inflated
    // slack by the element count.
    vector<int32_t> children;
    if (c.type_ == Fill::UNIVERSE) {
      children.push_back(c.fill_);
    } else if (c.type_ == Fill::LATTICE) {
      const Lattice& lat = *model::lattices[c.fill_];
      children.assign(lat.universes_.begin(), lat.universes_.end());
      if (lat.outer_ != NO_OUTER_UNIVERSE)
        children.push_back(lat.outer_);
      std::sort(children.begin(), children.end());
      children.erase(
        std::unique(children.begin(), children.end()), children.end());
    }
    for (int32_t i_child : children) {
      const Reach& cr = reach(i_child);
      r.slots.insert(r.slots.end(), cr.slots.begin(), cr.slots.end());
      r.pairs.insert(r.pairs.end(), cr.pairs.begin(), cr.pairs.end());
    }
  }

  std::sort(r.slots.begin(), r.slots.end());
  r.slots.erase(std::unique(r.slots.begin(), r.slots.end()), r.slots.end());
  std::sort(r.pairs.begin(), r.pairs.end());
  r.pairs.erase(std::unique(r.pairs.begin(), r.pairs.end()), r.pairs.end());
  return r;
}

//==============================================================================
// Accumulation
//==============================================================================

int OctreeVolumeCalculation::instance_of(
  const Cell& c, const vector<PathEntry>& path) const
{
  if (c.distribcell_index_ == C_NONE)
    return 0;
  int dc = c.distribcell_index_;
  int instance = 0;
  for (const auto& e : path) {
    const Cell& pc = *model::cells[e.i_cell];
    if (pc.type_ == Fill::UNIVERSE) {
      instance += pc.offset_[dc];
    } else if (pc.type_ == Fill::LATTICE) {
      instance += pc.offset_[dc];
      if (e.has_lat) {
        const Lattice& lat = *model::lattices[pc.fill_];
        // The array-indexed offset() overload is non-const, so go through the
        // flat index like resolve_lattice does.
        if (lat.are_valid_indices(e.lat_idx))
          instance += lat.offset(dc, lat.get_flat_index(e.lat_idx));
      }
    }
  }
  return instance;
}

void OctreeVolumeCalculation::credit(int32_t i_cell, int32_t i_univ,
  const vector<PathEntry>& path, double vol, Acc& acc) const
{
  const Cell& c = *model::cells[i_cell];

  // Which material actually fills this box. For a distributed material the
  // answer depends on the instance, so it has to be reconstructed from the
  // descent path -- splitting the volume evenly across material_ gives the
  // right total but the wrong per-material answer, which is what MATERIAL
  // domains report.
  int32_t i_mat = C_NONE;
  if (c.type_ == Fill::MATERIAL && !c.material_.empty()) {
    if (c.material_.size() == 1) {
      i_mat = c.material_[0];
    } else {
      int instance = instance_of(c, path);
      i_mat = (instance >= 0 && instance < static_cast<int>(c.material_.size()))
                ? c.material_[instance]
                : c.material_[0];
    }
  }

  auto add = [&](int slot) {
    acc.lower[slot] += vol;
    if (i_mat != C_NONE) {
      int p = find_pair(slot, i_mat);
      if (p >= 0)
        acc.mat_lower[p] += vol;
    }
  };

  auto ic = cell_slot_.find(i_cell);
  if (ic != cell_slot_.end())
    add(ic->second);

  auto iu = univ_slot_.find(i_univ);
  if (iu != univ_slot_.end())
    add(iu->second);

  if (i_mat != C_NONE) {
    auto im = mat_slot_.find(i_mat);
    if (im != mat_slot_.end())
      add(im->second);
  }
}

void OctreeVolumeCalculation::charge_slack(
  int32_t i_univ, double vol, Acc& acc) const
{
  const Reach& r = reach_[i_univ];
  for (int slot : r.slots)
    acc.slack[slot] += vol;
  for (int p : r.pairs)
    acc.mat_slack[p] += vol;
}

double OctreeVolumeCalculation::worst_half_width() const
{
  double w = 0.0;
  for (const auto& b : results_)
    w = std::max(w, b.half_width());
  return w;
}

void OctreeVolumeCalculation::fill_nuclides(int i_domain,
  vector<int>& nuclides, vector<double>& atoms,
  vector<double>& uncertainty) const
{
  // Mirrors the stochastic path's arithmetic: atom densities are in atoms/b-cm,
  // so atoms = 1e24 * V * density. What differs is the companion figure: the
  // volume here is an interval, so this is the half-width of the resulting
  // atom-count interval -- a hard bound, not a sigma.
  int n_nuc =
    settings::run_CE ? data::nuclides.size() : data::mg.nuclides_.size();
  vector<double> mean(n_nuc, 0.0);
  vector<double> halfw(n_nuc, 0.0);

  for (size_t p = 0; p < pair_mat_.size(); ++p) {
    if (pair_slot_[p] != i_domain)
      continue;
    int32_t i_material = pair_mat_[p];
    if (i_material == MATERIAL_VOID)
      continue;
    const auto& mat = model::materials[i_material];
    double v = mat_lower_[p] + 0.5 * mat_slack_[p];
    double dv = 0.5 * mat_slack_[p];
    for (int k = 0; k < mat->nuclide_.size(); ++k) {
      int i_nuclide = mat->nuclide_[k];
      mean[i_nuclide] += 1.0e24 * v * mat->atom_density_[k];
      halfw[i_nuclide] += 1.0e24 * dv * mat->atom_density_[k];
    }
  }

  for (int j = 0; j < n_nuc; ++j) {
    if (mean[j] > 0.0) {
      nuclides.push_back(j);
      atoms.push_back(mean[j]);
      uncertainty.push_back(halfw[j]);
    }
  }
}

//==============================================================================
// Lattice resolution
//==============================================================================
//
// Lattice elements -- rectangular prisms and hexagonal prisms alike -- are
// CONVEX, and a box is the convex hull of its eight corners. So if all eight
// corners land in the same element, the whole box does. That one argument
// covers both lattice types with no orientation conventions, no hexagon
// half-plane geometry, and no reliance on get_indices() being monotone in the
// corners, which for hex it is not. The failure mode is conservative: a corner
// exactly on a face may be assigned to the neighbour, the corners disagree, and
// the caller subdivides.

bool OctreeVolumeCalculation::resolve_lattice(const OctBox& box, int32_t i_lat,
  int32_t& i_child, OctBox& child_box, array<int, 3>& idx) const
{
  const Lattice& lat = *model::lattices[i_lat];
  Direction u {0.0, 0.0, 1.0}; // only used for boundary tie-breaking

  array<int, 3> other;
  lat.get_indices(box.corner(0), u, idx);
  for (int i = 1; i < 8; ++i) {
    lat.get_indices(box.corner(i), u, other);
    if (other != idx)
      return false;
  }

  if (lat.are_valid_indices(idx)) {
    // operator[] is non-const in Lattice, so go through the flat index.
    i_child = lat.universes_[lat.get_flat_index(idx)];
  } else {
    // Same element of the infinite tiling, but outside the lattice bounds.
    i_child = lat.outer_;
  }
  if (i_child == NO_OUTER_UNIVERSE)
    return false;

  // Element origin from the lattice's own coordinate map, so its indexing
  // arithmetic (hex skew bases included) never has to be reproduced here. The
  // centre is inside the element by the same convexity argument.
  Position origin = box.center - lat.get_local_position(box.center, idx);
  child_box = box.transformed(origin, vector<double> {});
  return true;
}

bool OctreeVolumeCalculation::try_integrate(const OctBox& box,
  const Universe& u, const std::unordered_map<int32_t, BoxSense>& memo,
  const vector<PathEntry>& path, Acc& acc) const
{
  vector<int32_t> strad;
  for (const auto& kv : memo)
    if (kv.second == BoxSense::BOTH)
      strad.push_back(kv.first);
  // The all-planes path costs 2^n polytope volumes, the curved paths 2^n
  // annulus evaluations; 8 is where that stops being worth it. Checked again
  // per path below, since the curved paths are the more expensive per piece.
  if (strad.empty() || strad.size() > 8)
    return false;
  std::sort(strad.begin(), strad.end()); // deterministic piece ordering
  int n = static_cast<int>(strad.size());

  auto sense_of = [&](int32_t i) {
    auto it = memo.find(i);
    return it == memo.end() ? BoxSense::UNKNOWN : it->second;
  };

  // Every cell still in play has to be a material fill: a piece of a box is not
  // a box, so there is nothing sensible to hand to a nested universe.
  for (int32_t i_cell : u.cells_) {
    if (!is_csg_[i_cell])
      return false;
    if (evaluate_region_tri(postfix_[i_cell], simple_[i_cell], sense_of) ==
        Tri::kFalse)
      continue;
    if (model::cells[i_cell]->type_ != Fill::MATERIAL)
      return false;
  }

  struct Piece {
    double vol;
    vector<BoxSense> signs; // parallel to strad
  };
  vector<Piece> pieces;
  double extra_slack = 0.0;

  //----------------------------------------------------------------------------
  // Sort the straddling surfaces.
  //----------------------------------------------------------------------------
  vector<Position> pl_n(n);
  vector<double> pl_d(n);
  vector<char> is_plane(n, 0);

  // Cylinders are grouped by (axis, centre). Within a group they are coaxial
  // and collapse to an annulus; groups have to be pairwise disjoint, which is
  // the paper's bundle-of-cylinders case.
  struct CylGroup {
    double c1, c2, r_max {0.0};
    vector<int> members;
  };
  vector<CylGroup> groups;
  int cyl_axis = -1;
  int sym = -1;
  AxisSymProfile prof;
  vector<int> sym_group;
  vector<double> sym_K;

  bool all_planes = true;
  for (int i = 0; i < n; ++i) {
    const auto& pr = proxies_[strad[i]];

    if (pr.kind == SurfaceProxy::Kind::QUADRIC &&
        as_plane(pr.q, pl_n[i], pl_d[i])) {
      is_plane[i] = 1;
      continue;
    }
    all_planes = false;

    // A rotated box is no longer fatal: box_axis_section below decides whether
    // its cross-section is constant along the surface's axis, which it is
    // whenever one box frame axis is parallel to that axis.
    int a;
    double c1, c2, r;
    if (pr.kind == SurfaceProxy::Kind::QUADRIC &&
        as_axis_cylinder(pr.q, a, c1, c2, r)) {
      if (cyl_axis < 0)
        cyl_axis = a;
      else if (a != cyl_axis)
        return false; // non-parallel cylinders
      int g = -1;
      for (size_t k = 0; k < groups.size(); ++k)
        if (groups[k].c1 == c1 && groups[k].c2 == c2)
          g = static_cast<int>(k);
      if (g < 0) {
        groups.push_back({c1, c2, 0.0, {}});
        g = static_cast<int>(groups.size()) - 1;
      }
      groups[g].members.push_back(i);
      groups[g].r_max = std::max(groups[g].r_max, r);
      continue;
    }

    if (pr.kind == SurfaceProxy::Kind::TORUS) {
      if (sym >= 0)
        return false; // a torus cannot join a concentric group
      prof = torus_profile(pr.ax, pr.tc, pr.A, pr.B, pr.C);
      sym = i;
      sym_group.push_back(i);
      sym_K.push_back(0.0);
      continue;
    }
    if (pr.kind == SurfaceProxy::Kind::QUADRIC) {
      // Concentric members agree on everything but the constant term, so they
      // collapse to an annulus exactly as coaxial cylinders do. This is the
      // case subdivision cannot rescue: two surfaces bounding a thin shell stay
      // mutually ambiguous over a 2-D region until the box edge drops below the
      // shell thickness, which for a 0.008 cm gap on a 0.4 cm pin is six extra
      // levels and ~4000x the boxes.
      AxisSymProfile cand;
      if (!as_axis_symmetric(pr.q, cand))
        return false;
      if (sym < 0) {
        prof = cand;
        sym = i;
      } else if (!same_revolved_geometry(prof, cand)) {
        return false;
      }
      sym_group.push_back(i);
      sym_K.push_back(cand.K);
      continue;
    }
    return false;
  }

  if (sym >= 0 && sym_group.size() > 1 &&
      prof.kind == AxisSymProfile::Kind::TORUS)
    return false; // a torus has no K to order a group by
  if (sym >= 0 && !groups.empty())
    return false; // a revolved surface and a cylinder bundle do not compose

  // Distinct cylinder groups must not overlap, or their sign combinations stop
  // being annuli.
  for (size_t a = 0; a < groups.size(); ++a) {
    for (size_t b = a + 1; b < groups.size(); ++b) {
      double dx = groups[a].c1 - groups[b].c1, dy = groups[a].c2 - groups[b].c2;
      if (std::sqrt(dx * dx + dy * dy) < groups[a].r_max + groups[b].r_max)
        return false;
    }
  }

  //----------------------------------------------------------------------------
  // Path 1: everything is a plane. Any orientation, any box, any count.
  //----------------------------------------------------------------------------
  if (all_planes) {
    for (int mask = 0; mask < (1 << n); ++mask) {
      vector<Position> nn(n);
      vector<double> dd(n);
      vector<BoxSense> signs(n);
      for (int i = 0; i < n; ++i) {
        bool negative = ((mask >> i) & 1) == 0;
        nn[i] = negative ? pl_n[i] : -1.0 * pl_n[i];
        dd[i] = negative ? pl_d[i] : -pl_d[i];
        signs[i] = negative ? BoxSense::NEGATIVE : BoxSense::POSITIVE;
      }
      double v;
      if (n == 1) {
        // The closed form is cheaper and better conditioned than the polytope.
        if (!box_halfspace_volume(box, nn[0], dd[0], v))
          return false;
      } else {
        v = box_planes_volume(box, nn, dd);
      }
      if (v > 0.0)
        pieces.push_back({v, signs});
    }
  } else {
    //--------------------------------------------------------------------------
    // Path 2: axis-aligned planes cut the box into sub-boxes; within each,
    // cylinder bundles give annuli and a revolved surface gives sliced bounds.
    //--------------------------------------------------------------------------
    if (n > 6)
      return false;

    int axis = (cyl_axis >= 0) ? cyl_axis : (sym >= 0 ? prof.axis : -1);
    if (axis < 0)
      return false;
    int r1 = (axis + 1) % 3, r2 = (axis + 2) % 3;

    // Work in the frame of the curved surface's axis rather than in sub-boxes.
    // The box contributes a constant cross-section polygon and an axial range;
    // planes then either clip the range or clip the polygon. This is both
    // simpler than splitting into sub-boxes and strictly more general, since a
    // box rotated ABOUT the axis still has a constant cross-section.
    vector<double> base_px, base_py;
    double w_lo, w_hi;
    if (!box_axis_section(box, axis, base_px, base_py, w_lo, w_hi))
      return false;

    auto comp = [](const Position& v, int k) {
      return k == 0 ? v.x : (k == 1 ? v.y : v.z);
    };

    // A plane is usable only if it is parallel to the axis (clips the polygon)
    // or perpendicular to it (clips the axial range). Anything in between makes
    // the cross-section vary along the axis.
    vector<int> axial_planes, radial_planes;
    for (int i = 0; i < n; ++i) {
      if (!is_plane[i])
        continue;
      double na = comp(pl_n[i], axis);
      double n1 = comp(pl_n[i], r1), n2 = comp(pl_n[i], r2);
      if (n1 == 0.0 && n2 == 0.0)
        axial_planes.push_back(i);
      else if (na == 0.0)
        radial_planes.push_back(i);
      else
        return false;
    }

    int na_pl = static_cast<int>(axial_planes.size());
    int nr_pl = static_cast<int>(radial_planes.size());
    int n_cyl = 0;
    for (const auto& g : groups)
      n_cyl += static_cast<int>(g.members.size());
    if ((1 << (na_pl + nr_pl + n_cyl)) > 512)
      return false;

    for (int am = 0; am < (1 << na_pl); ++am) {
      double lw = w_lo, hw = w_hi;
      vector<BoxSense> axial_signs(n, BoxSense::UNKNOWN);
      bool empty = false;
      for (int j = 0; j < na_pl; ++j) {
        int i = axial_planes[j];
        double na = comp(pl_n[i], axis);
        double value = pl_d[i] / na;
        bool below = na > 0.0; // negative halfspace is the low side
        bool take_low = ((am >> j) & 1) == 0;
        if (take_low)
          hw = std::min(hw, value);
        else
          lw = std::max(lw, value);
        if (hw <= lw) {
          empty = true;
          break;
        }
        axial_signs[i] =
          (take_low == below) ? BoxSense::NEGATIVE : BoxSense::POSITIVE;
      }
      if (empty)
        continue;
      double axial = hw - lw;

      for (int rm = 0; rm < (1 << nr_pl); ++rm) {
        vector<double> px = base_px, py = base_py;
        vector<BoxSense> signs = axial_signs;
        for (int j = 0; j < nr_pl; ++j) {
          int i = radial_planes[j];
          double a = comp(pl_n[i], r1), b = comp(pl_n[i], r2), c = pl_d[i];
          bool negative = ((rm >> j) & 1) == 0;
          if (negative) {
            clip_poly(px, py, a, b, c);
            signs[i] = BoxSense::NEGATIVE;
          } else {
            clip_poly(px, py, -a, -b, -c);
            signs[i] = BoxSense::POSITIVE;
          }
        }
        if (px.size() < 3)
          continue;
        double cross_area = std::abs(poly_area(px, py));
        if (cross_area <= 0.0)
          continue;
        double sub_vol = cross_area * axial;

        if (!groups.empty()) {
          for (int cm = 0; cm < (1 << n_cyl); ++cm) {
            vector<BoxSense> sg = signs;
            // Per group the "inside" constraints give the minimum radius and
            // the "outside" ones the maximum: the combination is an annulus
            // r_out < rho < r_in.
            int bit = 0, n_bounded = 0, i_bounded = -1;
            vector<double> g_in(groups.size(), INFTY), g_out(groups.size(), 0.0);
            bool dead = false;
            for (size_t g = 0; g < groups.size() && !dead; ++g) {
              for (int i : groups[g].members) {
                int aa;
                double d1, d2, r;
                as_axis_cylinder(proxies_[strad[i]].q, aa, d1, d2, r);
                if (((cm >> bit) & 1) == 0) {
                  g_in[g] = std::min(g_in[g], r);
                  sg[i] = BoxSense::NEGATIVE;
                } else {
                  g_out[g] = std::max(g_out[g], r);
                  sg[i] = BoxSense::POSITIVE;
                }
                ++bit;
              }
              if (g_out[g] >= g_in[g])
                dead = true;
              if (g_in[g] != INFTY) {
                ++n_bounded;
                i_bounded = static_cast<int>(g);
              }
            }
            if (dead || n_bounded > 1)
              continue; // disjoint groups cannot both contain the point

            double area;
            if (n_bounded == 1) {
              const auto& g = groups[i_bounded];
              area = circle_poly_area(g.c1, g.c2, g_in[i_bounded], px, py) -
                     circle_poly_area(g.c1, g.c2, g_out[i_bounded], px, py);
            } else {
              area = cross_area;
              for (size_t g = 0; g < groups.size(); ++g)
                area -= circle_poly_area(
                  groups[g].c1, groups[g].c2, g_out[g], px, py);
            }
            if (area <= 0.0)
              continue;
            pieces.push_back({area * axial, sg});
          }
        } else if (sym >= 0) {
          // Bound each member's disk integral ONCE, then build every sign
          // combination by differencing. A combination is the annulus between
          // the binding "inside" surface (largest K, smallest radius) and the
          // binding "outside" one (smallest K, largest radius), and since
          // integral bounds subtract exactly the way per-slice bounds do, this
          // gives the identical bracket for ns evaluations instead of 2^ns.
          int ns = (int)sym_group.size();
          bool torus = prof.kind == AxisSymProfile::Kind::TORUS;
          double target = tolerance_ * sub_vol / root_.volume();
          vector<double> mlo(ns), mhi(ns);
          for (int j = 0; j < ns; ++j) {
            AxisSymProfile pj = prof;
            if (!torus)
              pj.K = sym_K[j];
            int N = 32;
            axis_sym_slice_bounds_poly(pj, lw, hw, px, py, N, mlo[j], mhi[j]);
            double spread = mhi[j] - mlo[j];
            if (spread > target && target > 0.0) {
              double want = 32.0 * spread / target;
              while (N < want && N < max_slices_)
                N *= 2;
              axis_sym_slice_bounds_poly(pj, lw, hw, px, py, N, mlo[j], mhi[j]);
            }
          }

          double sum_lo = 0.0;
          for (int sm = 0; sm < (1 << ns); ++sm) {
            int ia = -1, ib = -1;
            vector<BoxSense> sg = signs;
            for (int j = 0; j < ns; ++j) {
              bool inside = ((sm >> j) & 1) == 0;
              sg[sym_group[j]] =
                inside ? BoxSense::NEGATIVE : BoxSense::POSITIVE;
              if (inside) {
                if (ia < 0 || sym_K[j] > sym_K[ia])
                  ia = j;
              } else {
                if (ib < 0 || sym_K[j] < sym_K[ib])
                  ib = j;
              }
            }
            if (ia >= 0 && ib >= 0 && sym_K[ia] >= sym_K[ib])
              continue; // inner radius would exceed the outer: empty
            double a_lo = (ia >= 0) ? mlo[ia] : sub_vol;
            double a_hi = (ia >= 0) ? mhi[ia] : sub_vol;
            double b_lo = (ib >= 0) ? mlo[ib] : 0.0;
            double b_hi = (ib >= 0) ? mhi[ib] : 0.0;
            double slo = a_lo - b_hi;
            if (slo > 0.0) {
              pieces.push_back({slo, sg});
              sum_lo += slo;
            }
          }
          // Whatever the bounds failed to account for is slack, which also
          // keeps the tiling check below exact.
          extra_slack += sub_vol - sum_lo;
        } else {
          pieces.push_back({sub_vol, signs});
        }
      }
    }
  }

  if (pieces.empty())
    return false;

  // The pieces must tile the box. This is a cheap, strong invariant on every
  // path -- a bug in the sign enumeration, the disjointness reasoning or the
  // polytope clipper shows up here rather than as a quietly wrong volume.
  double sum = 0.0;
  for (const auto& pc : pieces)
    sum += pc.vol;
  if (std::abs(sum + extra_slack - box.volume()) > 1.0e-9 * box.volume())
    return false;

  // Resolve each piece to the one cell that owns it, in a dry pass first:
  // bailing out halfway would leave partial credit behind.
  vector<int32_t> owner(pieces.size(), C_NONE);
  for (size_t p = 0; p < pieces.size(); ++p) {
    auto definite = [&](int32_t i) -> BoxSense {
      for (int k = 0; k < n; ++k)
        if (strad[k] == i)
          return pieces[p].signs[k];
      return sense_of(i);
    };
    int hits = 0;
    for (int32_t i_cell : u.cells_) {
      if (evaluate_region_tri(postfix_[i_cell], simple_[i_cell], definite) ==
          Tri::kTrue) {
        owner[p] = i_cell;
        ++hits;
      }
    }
    if (hits > 1)
      return false; // overlapping cells: let subdivision surface the problem
  }

  int32_t i_univ = model::universe_map.at(u.id_);
  for (size_t p = 0; p < pieces.size(); ++p)
    if (owner[p] != C_NONE)
      credit(owner[p], i_univ, path, pieces[p].vol, acc);

  charge_slack(i_univ, extra_slack + INTEGRATOR_GUARD * box.volume(), acc);
  return true;
}

//==============================================================================
// Traversal
//==============================================================================

int OctreeVolumeCalculation::process(OctBox& box, int32_t& i_univ, int depth,
  int max_depth, vector<PathEntry>& path, Acc& acc, OctBox* kids) const
{
  while (true) {
    const Universe& u = *model::universes[i_univ];

    // Per-box memo so a surface shared by several cells in this universe is
    // classified once.
    std::unordered_map<int32_t, BoxSense> memo;
    auto sense = [&](int32_t i_surf) -> BoxSense {
      auto it = memo.find(i_surf);
      if (it != memo.end())
        return it->second;
      BoxSense s = this->classify(i_surf, box);
      memo.emplace(i_surf, s);
      return s;
    };

    int n_true = 0;
    int32_t i_true = C_NONE;
    bool any_maybe = false;
    bool lattice_spanning = false;

    for (int32_t i_cell : u.cells_) {
      if (!is_csg_[i_cell]) {
        any_maybe = true; // DAGMC cell: cannot resolve, force refinement
        continue;
      }
      Tri t = evaluate_region_tri(postfix_[i_cell], simple_[i_cell], sense);
      if (t == Tri::kTrue) {
        ++n_true;
        i_true = i_cell;
      } else if (t == Tri::kMaybe) {
        any_maybe = true;
      }
    }

    if (n_true == 0 && !any_maybe)
      return 0; // box lies outside every cell of this universe

    // The box lies wholly inside exactly one cell. This is the only place a
    // frame change happens, and it is safe precisely because containment is
    // already proven: the transformed box is the same physical box, so no
    // volume is created or lost and sibling boxes still tile.
    if (n_true == 1 && !any_maybe) {
      const Cell& c = *model::cells[i_true];

      if (c.type_ == Fill::MATERIAL) {
        credit(i_true, i_univ, path, box.volume(), acc);
        return 0;
      }

      if (c.type_ == Fill::UNIVERSE) {
        box = box.transformed(c.translation_, c.rotation_);
        path.push_back({i_true, {0, 0, 0}, false});
        i_univ = c.fill_;
        continue; // same box, one level down
      }

      if (c.type_ == Fill::LATTICE) {
        // geometry.cpp applies the cell translation and rotation before
        // indexing the lattice, so do the same here.
        OctBox lb = box.transformed(c.translation_, c.rotation_);
        int32_t i_child;
        OctBox child_box;
        array<int, 3> idx;
        if (resolve_lattice(lb, c.fill_, i_child, child_box, idx)) {
          box = child_box;
          path.push_back({i_true, idx, true});
          i_univ = i_child;
          continue;
        }
        // Box spans several elements, or is outside a lattice with no outer
        // universe: subdivide, but on all three axes -- the lattice boundary
        // is not one of the surfaces the mask below reasons about.
        lattice_spanning = true;
      }
    }

    if (n_true > 1) {
      // Reported once after the sweep rather than here: this runs inside a
      // parallel region and could fire on millions of boxes.
      overlap_universe_.store(u.id_, std::memory_order_relaxed);
    }

    // Try to finish this box in closed form before spending more depth on it.
    if (try_integrate(box, u, memo, path, acc))
      return 0;

    if (depth >= max_depth) {
      charge_slack(i_univ, box.volume(), acc);
      return 0;
    }

    // Pick the axes worth splitting. If no straddling surface's classification
    // varies along an axis, the two children come out identical and the split
    // is pure waste. Reactor geometry is overwhelmingly extruded, so this is
    // not a corner case: on a pincell it takes the node count per level from
    // x4.03 to x2.13, which is the difference between eps^-2 and eps^-1.
    int mask = 0;
    if (lattice_spanning) {
      mask = 0b111;
    } else {
      for (int k = 0; k < 3; ++k) {
        for (const auto& kv : memo) {
          if (kv.second != BoxSense::BOTH)
            continue;
          const auto& pr = proxies_[kv.first];
          // Only quadrics expose the structure needed to prove irrelevance;
          // anything else conservatively forces the split.
          if (pr.kind != SurfaceProxy::Kind::QUADRIC ||
              !axis_irrelevant(pr.q, box, k, form_gradient(pr.q, box))) {
            mask |= (1 << k);
            break;
          }
        }
      }
      if (mask == 0)
        mask = 0b111; // nothing straddles; should not happen, stay safe
    }

    int n = OctBox::n_children(mask);
    for (int i = 0; i < n; ++i)
      kids[i] = box.child(i, mask);
    return n;
  }
}

void OctreeVolumeCalculation::descend(OctBox box, int32_t i_univ, int depth,
  int max_depth, vector<PathEntry>& path, Acc& acc) const
{
  size_t mark = path.size();
  OctBox kids[8];
  int n = process(box, i_univ, depth, max_depth, path, acc, kids);
  for (int i = 0; i < n; ++i)
    descend(kids[i], i_univ, depth + 1, max_depth, path, acc);
  path.resize(mark);
}

void OctreeVolumeCalculation::build_frontier(
  int max_depth, Acc& serial, vector<Node>& frontier) const
{
  // Grow the work list breadth-first, only as far as it takes to keep the
  // workers busy. A fixed 8^k pre-split wasted 512 boxes on models that
  // resolve at the root, and actively hurt the revolved-surface integrator,
  // whose cost for a target slack goes as A^2/(h*s) -- so N slices are needed
  // per box regardless of size, and 512 boxes cost 512 times as much as one.
  int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif
  size_t target = static_cast<size_t>(
    std::max(1, tasks_per_worker_ * n_threads * std::max(1, mpi::n_procs)));

  vector<Node> current;
  current.push_back({root_, model::root_universe, 0, {}});

  while (current.size() < target) {
    vector<Node> next;
    bool expanded = false;
    for (auto& node : current) {
      OctBox box = node.box;
      int32_t i_univ = node.i_univ;
      vector<PathEntry> path = node.path;
      OctBox kids[8];
      int n = process(box, i_univ, node.depth, max_depth, path, serial, kids);
      if (n == 0)
        continue; // fully handled during construction
      expanded = true;
      for (int i = 0; i < n; ++i)
        next.push_back({kids[i], i_univ, node.depth + 1, path});
    }
    if (!expanded)
      return; // nothing left to subdivide; everything went into `serial`
    current = std::move(next);
    if (current.empty())
      return;
  }
  frontier = std::move(current);
}

void OctreeVolumeCalculation::sweep(int max_depth)
{
  std::fill(lower_.begin(), lower_.end(), 0.0);
  std::fill(slack_.begin(), slack_.end(), 0.0);
  std::fill(mat_lower_.begin(), mat_lower_.end(), 0.0);
  std::fill(mat_slack_.begin(), mat_slack_.end(), 0.0);
  overlap_universe_.store(C_NONE, std::memory_order_relaxed);

  size_t n_slots = lower_.size(), n_pairs = mat_lower_.size();

  // The frontier is built identically on every rank, so only one of them may
  // keep the work done along the way or it would be counted n_procs times.
  Acc serial;
  serial.init(n_slots, n_pairs);
  vector<Node> frontier;
  build_frontier(max_depth, serial, frontier);
  if (mpi::master) {
    for (size_t j = 0; j < n_slots; ++j) {
      lower_[j] += serial.lower[j];
      slack_[j] += serial.slack[j];
    }
    for (size_t j = 0; j < n_pairs; ++j) {
      mat_lower_[j] += serial.mat_lower[j];
      mat_slack_[j] += serial.mat_slack[j];
    }
  }

  // Static round-robin over ranks. Frontier nodes are not equal-cost, so the
  // threads within a rank use a dynamic schedule to even that out.
  vector<int> mine;
  for (int i = mpi::rank; i < static_cast<int>(frontier.size());
       i += mpi::n_procs)
    mine.push_back(i);
  int n_mine = static_cast<int>(mine.size());

#pragma omp parallel
  {
    Acc local;
    local.init(n_slots, n_pairs);
    vector<PathEntry> path;

#pragma omp for schedule(dynamic)
    for (int i = 0; i < n_mine; ++i) {
      const Node& node = frontier[mine[i]];
      path = node.path;
      descend(node.box, node.i_univ, node.depth, max_depth, path, local);
    }

    // Merge with atomics rather than a critical section. The updates
    // themselves stay thread-local: an atomic per credit would mean millions
    // of synchronisations against a handful here, which is why the
    // accumulators are dense but private.
    for (size_t j = 0; j < n_slots; ++j) {
      if (local.lower[j] != 0.0) {
#pragma omp atomic
        lower_[j] += local.lower[j];
      }
      if (local.slack[j] != 0.0) {
#pragma omp atomic
        slack_[j] += local.slack[j];
      }
    }
    for (size_t j = 0; j < n_pairs; ++j) {
      if (local.mat_lower[j] != 0.0) {
#pragma omp atomic
        mat_lower_[j] += local.mat_lower[j];
      }
      if (local.mat_slack[j] != 0.0) {
#pragma omp atomic
        mat_slack_[j] += local.mat_slack[j];
      }
    }
  }

#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    MPI_Allreduce(MPI_IN_PLACE, lower_.data(), lower_.size(), MPI_DOUBLE,
      MPI_SUM, mpi::intracomm);
    MPI_Allreduce(MPI_IN_PLACE, slack_.data(), slack_.size(), MPI_DOUBLE,
      MPI_SUM, mpi::intracomm);
    if (!mat_lower_.empty()) {
      MPI_Allreduce(MPI_IN_PLACE, mat_lower_.data(), mat_lower_.size(),
        MPI_DOUBLE, MPI_SUM, mpi::intracomm);
      MPI_Allreduce(MPI_IN_PLACE, mat_slack_.data(), mat_slack_.size(),
        MPI_DOUBLE, MPI_SUM, mpi::intracomm);
    }
  }
#endif

  for (size_t j = 0; j < n_slots; ++j) {
    results_[j].lower = lower_[j];
    results_[j].slack = slack_[j];
  }
  depth_reached_ = max_depth;

  int overlap = overlap_universe_.load(std::memory_order_relaxed);
  if (overlap != C_NONE) {
    warning(fmt::format("Octree volume calculation found overlapping cells in "
                        "universe {}; results are not trustworthy.",
      overlap));
  }
}

void OctreeVolumeCalculation::run()
{
  // Iterative deepening, driven by the slack we actually measure.
  //
  // The obvious alternative -- pick a minimum box volume up front from the
  // tolerance -- cannot work, because the depth a tolerance needs depends on
  // the surface area of the domain boundary, which is exactly the quantity
  // nobody knows in advance.
  //
  // Deepening costs at most one extra sweep's worth of work, since cost per
  // level grows by at least 2x, and the stopping test is then a fact rather
  // than an estimate.
  for (int d = std::max(start_depth_, 1); d <= max_depth_; ++d) {
    sweep(d);
    if (worst_half_width() <= tolerance_)
      return;
  }
}

} // namespace openmc
