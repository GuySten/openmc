//! \file bep.cpp
//! Branched Exact Perturbation. See bep.h for the estimator and lineage.

#include "openmc/bep.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include <fmt/core.h>

#include "openmc/cell.h"
#include "openmc/constants.h"
#include "openmc/container_util.h"
#include "openmc/error.h"
#include "openmc/file_utils.h"
#include "openmc/material.h"
#include "openmc/message_passing.h"
#include "openmc/openmp_interface.h"
#include "openmc/mgxs_interface.h"
#include "openmc/bank.h"
#include "openmc/particle.h"
#include "openmc/particle_data.h"
#include "openmc/physics.h"
#include "openmc/reaction.h"
#include "openmc/random_lcg.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/tallies/tally.h"
#include "openmc/xml_interface.h"

namespace openmc {

// Declared in simulation.h; repeated to avoid a circular include.
void transport_history_based_single_particle(Particle& p);

namespace bep {

vector<Perturbation> perturbations;
vector<int> tree_pert;
vector<int> tree_class;
vector<vector<int>> cell_perts;
vector<double> tree_site_weight;
vector<double> tree_weight_scale;
vector<int64_t> tree_sources;
vector<int64_t> batch_sources;
vector<double> batch_root_sq;
vector<double> thread_root_sq;
vector<double> root_x;
vector<double> batch_pair_sq;
vector<int64_t> batch_pair_n;
int64_t n_history_total {0};
int64_t batch_histories {0};
vector<vector<TrackSite>> thread_tracks;
vector<TrackSite> tracks;
vector<vector<SourceRoot>> thread_source_roots;
vector<SourceRoot> source_roots;
vector<double> thread_tau;
vector<double> thread_root_weight;
vector<double> tau;
vector<double> batch_tau;
vector<double> pooled_tau;
vector<double> ell_sum;
vector<double> ell_cross;
int64_t n_active_batches {0};
int64_t n_generations {0};

// Internal linkage. `inline` would be redundant inside an unnamed namespace,
// so it is omitted; forward declarations let run_one_tree() live here with
// the rest of the file-local helpers rather than in a second block further
// down.
namespace {

//! Transport one shadow tree rooted at `site` in tree `tree`.
//!
//! Depth bookkeeping rides on the existing super-history fields:
//! create_fission_sites() stamps super_gen on each site it makes and calls
//! score_site() with it, so nothing has to be measured after the fact.
//!
//! `seed_id` is shared by the trees that have to cancel against each other --
//! the two halves of a +/- pair, and the same source in two perturbations --
//! which is what makes them common-random-number correlated. ALL streams must be
//! initialised from it, not just STREAM_TRACKING: ParticleData's constructor
//! leaves seeds_ uninitialised and from_source() does not touch it, so a
//! partially seeded particle picks up stack garbage for the other streams.
//! STREAM_URR_PTABLE in particular is advanced on every energy change in
//! sample_neutron_reaction(), so leaving it unseeded silently decorrelates
//! the trees wherever a nuclide has unresolved resonances -- and a null
//! perturbation then returns noise instead of zero.
void run_one_tree(const SourceRoot& site, double* x_out)
{
  const int tree = site.tree;
  Particle p;

  SourceSite root;
  root.r = site.r;
  root.u = site.u;
  root.E = site.E;
  root.wgt = site.wgt;
  root.time = site.time;
  root.particle = ParticleType::neutron();
  root.super_gen = 1; // see the depth-encoding note in bep.h
  root.adjoint_id = -1;
  root.bep_tree = tree;

  p.from_source(&root);
  p.n_progeny() = 0;
  p.n_event() = 0;
  p.n_tracks() = 1;
  p.n_split() = 0;
  p.ww_factor() = 0.0;
  // wgt_born is the scale apply_russian_roulette() measures against, and for
  // a shadow tree that must be the weight a typical particle in the tree is
  // born at, not the weight of the root it grew from. Those differ by
  // how far below its reference the perturbation's population actually sits:
  // judged against the root weight, such a tree's entire population is below
  // the cutoff and survival biasing would roulette all of it up to
  // weight_survive. weight_scale(), NOT site_weight() -- the latter carries a
  // 1/perturbation_site_splitting's chosen population, so using it here would
  // move the roulette
  // threshold every time the population knob was tuned. weight_scale() is 1.0
  // for a reference tree and for a material perturbation's tree, so this is
  // p.wgt() exactly for them and they are bit-identical to before it
  // existed.
  //
  // Read only by apply_russian_roulette(), which returns immediately unless
  // survival_biasing is on, so with it off this changes nothing anywhere.
  p.wgt_born() = p.wgt() * weight_scale(tree);
  p.id() = site.seed_id;
  init_particle_seeds(site.seed_id, p.seeds());
  p.stream() = STREAM_TRACKING;

  // The scale everything inside this tree is measured against; see
  // root_weight(). Set before transport and cleared after, so that nothing
  // outside a shadow tree can read a stale value.
  thread_root_weight[thread_num()] = site.wgt;

  // One independent source event for this tree: the M of the variance law,
  // counted rather than estimated.
#pragma omp atomic
  tree_sources[tree] += 1;

  score_site(tree, 1, site.wgt); // the root is this tree's depth-0 weight

  // This ROOT's own depth-L descendant weight, isolated by bracketing its
  // transport. The roots are independent samples of exactly the quantity
  // c_t describes, so accumulating the square here measures c_t directly
  // instead of leaving it to be subtracted out of the measured spread.
  const size_t slab = static_cast<size_t>(thread_num()) * tau_stride();
  const size_t cell_L =
    slab + static_cast<size_t>(tau_index(tree, settings::bep_n_generation));
  const double before = thread_tau[cell_L];

  transport_history_based_single_particle(p);

  const double x = thread_tau[cell_L] - before;
  // run_one_tree() set n_tracks() = 1 for the root and
  // event_revive_from_secondary() increments it once per revival, so
  // n_tracks() IS this tree's history count. Counted, never timed.
#pragma omp atomic
  n_history_total += p.n_tracks();
#pragma omp atomic
  batch_histories += p.n_tracks();
  thread_root_sq[static_cast<size_t>(thread_num()) * tree_pert.size() +
                 static_cast<size_t>(tree)] += x * x;

  // This root's own depth-L contribution, kept so that the SIGNED combination
  // over a common-random-number group can be formed after the parallel loop.
  // Writing it per root and grouping serially afterwards keeps the grouping
  // out of the parallel region: a pair is adjacent in the sorted root list but
  // a static schedule can still split it across two threads.
  if (x_out)
    *x_out = x;

  p.local_secondary_bank().clear();
  thread_root_weight[thread_num()] = 0.0;
}

//! The reference eigenvalue the fission bank was normalised by, and hence the
//! kk of level(). Captured at the start of the shadow pass so that every
//! reader within a generation uses one value.
double keff_norm {1.0};

//! Evaluate a material's macroscopic cross sections at a track segment's
//! phase point, on a scratch particle.
//!
//! Done here, in the shadow pass, and NOT at the driver collision that
//! recorded the segment. Evaluating a second material on the driver itself
//! would overwrite its cached macro cross sections and could draw from its
//! unresolved-resonance stream, and the driver has to stay bit-identical to a
//! stock run. `mat` may be MATERIAL_VOID, which has no cross sections at all.
void material_macro(Particle& s, const TrackSite& t, int32_t mat, double& st,
  double& sa, double& snf)
{
  if (mat == MATERIAL_VOID) {
    st = sa = snf = 0.0;
    return;
  }
  s.E() = t.E;
  s.sqrtkT() = t.sqrtkT;
  s.density_mult() = t.density_mult;
  model::materials[mat]->calculate_xs(s);
  st = s.macro_xs().total;
  sa = s.macro_xs().absorption;
  snf = s.macro_xs().nu_fission;
}

//! One nuclide of a material, as the source generator sees it.
struct NuclideEntry {
  int i_nuclide;
  int index_sab;
  double sab_frac;
  double density;
};

//! List the nuclides of `mat` at this phase point, with `mat`'s own thermal
//! scattering assignment. The scratch particle must already hold `mat`'s
//! cross sections.
void list_nuclides(
  Particle& s, const TrackSite& t, int32_t mat, vector<NuclideEntry>& out)
{
  out.clear();
  if (mat == MATERIAL_VOID)
    return;
  const Material& m {*model::materials[mat]};
  for (int i = 0; i < static_cast<int>(m.nuclides().size()); ++i) {
    int i_nuc = m.nuclides()[i];
    const auto& micro = s.neutron_xs(i_nuc);
    out.push_back({i_nuc, micro.index_sab, micro.sab_frac,
      m.atom_density(i, t.density_mult)});
  }
}

//! Emit this nuclide's contribution to the perturbation source.
//!
//! `d_density` is the SIGNED change in this nuclide's atom density that the
//! substitution makes -- zero for every nuclide the substitution leaves
//! alone, which is why nothing is emitted for them at all. That is the whole
//! point of decomposing the source by nuclide rather than by material: a
//! material-level pair, +Sigma_s' from one kernel and -Sigma_s from the
//! other, would emit two O(Sigma_s) populations whose difference is the
//! answer, and the noise would be set by the populations rather than by the
//! difference. Measured on a water-plus-boron substitution, that cost a
//! factor of order the ratio between them.
//!
//! The scratch particle must hold the cross sections of the material this
//! nuclide was listed from, and s.material() must be that material:
//! scatter() reads its isotropic-in-lab flags through it.
void emit_nuclide_source(Particle& s, const TrackSite& t,
  const NuclideEntry& e, double d_density, const Perturbation& pert,
  const Position& r_emit, double base, int64_t site_seed,
  vector<SourceRoot>& out)
{
  if (d_density == 0.0)
    return;
  const auto& micro = s.neutron_xs(e.i_nuclide);
  const double w = std::abs(d_density) * base;
  const bool positive = d_density > 0.0;

  // A root whose outgoing state is SAMPLED must not then be grown on the
  // stream that sampled it.
  //
  // Both the fission root and the in-scatter root (b) below draw their (E, u)
  // from a random stream, and each used to carry that same stream as its
  // seed_id -- so run_one_tree()'s init_particle_seeds(site.seed_id, ...)
  // replayed, as the tree's transport, the very numbers that had produced the
  // tree's own starting energy. That is not a variance-reduction trick, it is
  // a bias: writing the root's state as E = h(xi_1..xi_k) and the walk as
  // g(xi_1, xi_2, ...), the estimator converges to E[g(xi; h(xi))] and not to
  // E_E[ E_xi'[ g(xi'; E) ] ] = <phi', S>. The walk has to be conditionally
  // independent of the phase point it starts from, and sharing a stream is
  // exactly the dependence that breaks it.
  //
  // Roots (a) and (c) are untouched by this: their phase point is the driver
  // segment's own (E, u), sampled by nothing, so no stream generated it.
  //
  // The fix keeps the (b)/(c) common-random-numbers pairing that the
  // decomposition below relies on -- both still share ONE tree seed -- and
  // only makes that tree seed different from the sampling seed. Keys 2 and 3
  // sample; keys 4, 5 and 6 grow.
  const int64_t fission_tree_seed =
    combine_ids({site_seed, e.i_nuclide, 5});
  const int64_t pair_tree_seed = combine_ids({site_seed, e.i_nuclide, 6});

  // ---- fission production: + chi_i nu sigma_f,i -------------------------
  if (micro.nu_fission > 0.0) {
    init_particle_seeds(
      combine_ids({site_seed, e.i_nuclide, 2}), s.seeds());
    s.stream() = STREAM_TRACKING;
    s.E() = t.E;
    s.u() = t.u;
    SourceSite site;
    site.time = t.time;
    const Reaction& rx = sample_fission(e.i_nuclide, s);
    sample_fission_neutron(e.i_nuclide, rx, &site, s);

    SourceRoot root;
    root.r = r_emit;
    root.u = site.u;
    root.E = site.E;
    root.time = site.time;
    root.wgt = w * micro.nu_fission;
    root.tree = positive ? pert.tree_fp : pert.tree_fn;
    root.seed_id = fission_tree_seed;
    out.push_back(root);
  }

  // ---- removal and in-scatter ------------------------------------------
  //
  // This nuclide's share of -(dL psi) is
  //
  //     -dn_i sigma_t,i psi(E,u)  +  dn_i sigma_s,i psi(E,u) scattered
  //
  // and writing it that way directly is a disaster: sigma_t and sigma_s
  // differ only by the absorption, so the answer is a difference of two
  // nearly equal populations. Measured on a dissolved-U235 substitution the
  // two came to 4530 and 4450 against a true difference of order 50 -- a
  // hundredfold cancellation, with the level wandering by its own size from
  // one depth to the next.
  //
  // Splitting sigma_t = sigma_a + sigma_s first gives three roots instead,
  //
  //     (a)  -dn_i sigma_a,i   at (E, u)              the absorption
  //     (b)  +dn_i sigma_s,i   at the SCATTERED state
  //     (c)  -dn_i sigma_s,i   at (E, u)              unscattered
  //
  // which is the same thing exactly, but now (a) is a clean one-signed term
  // carrying the absorption -- usually most of the worth -- and the whole of
  // the remaining cancellation sits in (b) + (c), whose weights are
  // IDENTICAL by construction. That pair is the importance change across one
  // scattering event; sharing its seed makes the two trees track each other
  // wherever they are doing the same physics, and with equal weights there
  // is nothing left to spoil the cancellation but the scattering itself.
  //
  // What that costs is honest and irreducible here: for a scatterer that
  // throws the neutron somewhere quite different -- hydrogen above all --
  // the two trees decorrelate at once and the noise is set by sigma_s rather
  // than by the answer. That is a property of a signed source, not of this
  // bookkeeping, and it is why the per-nuclide decomposition above matters
  // so much: a nuclide the substitution leaves alone emits none of the
  // three.
  double sigma_a = micro.absorption;
  double sigma_s = micro.total - micro.absorption;
  // Key 3 SAMPLES the scattered state; pair_tree_seed (key 6) GROWS both
  // members of the pair. See the note above on why those must differ.
  int64_t seed = combine_ids({site_seed, e.i_nuclide, 3});

  // (a) absorption, at the segment's own phase point
  if (sigma_a > 0.0) {
    SourceRoot root;
    root.r = r_emit;
    root.u = t.u;
    root.E = t.E;
    root.time = t.time;
    root.wgt = w * sigma_a;
    root.tree = positive ? pert.tree_ln : pert.tree_lp; // note: -dn
    root.seed_id = combine_ids({site_seed, e.i_nuclide, 4});
    out.push_back(root);
  }

  if (sigma_s > 0.0) {
    // (c) the unscattered member of the pair
    SourceRoot before;
    before.r = r_emit;
    before.u = t.u;
    before.E = t.E;
    before.time = t.time;
    before.wgt = w * sigma_s;
    before.tree = positive ? pert.tree_ln : pert.tree_lp; // note: -dn
    before.seed_id = pair_tree_seed;
    out.push_back(before);

    // (b) the scattered member: same weight, and the same TREE seed as (c)
    // so the pair still tracks, but sampled from a different stream.
    init_particle_seeds(seed, s.seeds());
    s.stream() = STREAM_TRACKING;
    s.E() = t.E;
    s.u() = t.u;
    s.wgt() = 1.0;
    s.local_secondary_bank().clear();

    scatter(s, e.i_nuclide);

    SourceRoot root;
    root.r = r_emit;
    root.u = s.u();
    root.E = s.E();
    root.time = t.time;
    root.wgt = w * sigma_s * s.wgt();
    root.tree = positive ? pert.tree_lp : pert.tree_ln;
    root.seed_id = pair_tree_seed;
    if (root.wgt > 0.0)
      out.push_back(root);

    // The multiplicity of an (n,xn) comes out of scatter() too -- as a
    // weight factor for a non-integral yield, as extra neutrons in the local
    // bank for an integral one -- and both are carried, because this is a
    // source of NEUTRONS and an (n,2n) makes two.
    for (const auto& sec : s.local_secondary_bank()) {
      SourceRoot extra = root;
      extra.u = sec.u;
      extra.E = sec.E;
      extra.wgt = w * sigma_s * sec.wgt;
      if (extra.wgt > 0.0)
        out.push_back(extra);
    }
    s.local_secondary_bank().clear();
  }
}

//! Emit the perturbation source dH psi carried by one driver track segment.
//!
//! Track-length form: the expected source from a segment of length l carrying
//! weight w is (dSigma * w * l), with the emission point uniform along the
//! segment. Three channels make up  -(dL psi) = -dSigma_t psi + dSigma_s psi
//! and  dF psi = d(chi nu Sigma_f) psi:
//!
//!   removal    -(Sigma_t' - Sigma_t) * w * l   at the segment's own (u, E)
//!   in-scatter  +dn_i * sigma_s,i * w * l      per nuclide, from ITS kernel
//!   fission     +dn_i * nu sigma_f,i * w * l   per nuclide, from ITS spectrum
//!
//! The removal channel is a single signed root because both sides share a
//! phase point exactly, so what is emitted is already the small difference.
//! The other two cannot be written that way -- the outgoing distributions
//! differ -- so they are decomposed by NUCLIDE instead, and each nuclide
//! contributes in proportion to the change dn_i in its atom density. A
//! nuclide the substitution leaves alone has dn_i = 0 and emits nothing at
//! all, so a null substitution produces no source whatsoever and the worth is
//! a hard zero rather than the difference of two noisy estimates.
//!
//! A nuclide whose THERMAL SCATTERING assignment differs between the two
//! materials is not the same scatterer in both, so it is not matched: it
//! contributes in full on each side, and only those two populations have to
//! cancel against each other.
void emit_perturbation_source(
  const TrackSite& t, Particle& s, vector<SourceRoot>& out)
{
  const double base = t.wgt * t.distance;
  if (base <= 0.0)
    return;

  // Seed the scratch particle before anything reads a random number from it.
  // Evaluating a material's cross sections can draw from the
  // unresolved-resonance stream, and a Particle's seeds are not initialised
  // by its constructor -- an unseeded scratch would take stack garbage and
  // make the source non-reproducible.
  init_particle_seeds(t.seed_id, s.seeds());
  s.stream() = STREAM_TRACKING;
  s.n_secondaries() = 0;
  s.r() = t.r;
  s.time() = t.time;
  s.local_secondary_bank().clear();

  vector<NuclideEntry> ref_nuc, pert_nuc;

  for (int ip : cell_perts[t.cell]) {
    int32_t mat_p;
    if (!substitute(ip, t.cell, mat_p))
      continue; // this perturbation touches the cell only through another sub
    const Perturbation& pert = perturbations[ip];

    // The emission point, shared by every channel of every perturbation on
    // this segment. Deliberately NOT keyed on `ip`: two perturbations that
    // substitute into the same cell should emit the same nuclide's source at
    // the same place, from the same outgoing state, differing only in the
    // weight -- their trees are then the same tree and almost all of their
    // noise is common. That shared noise is what makes the DIFFERENCE
    // between two perturbations far better determined than either one, which
    // is the whole reason for running them together. Keying the seed on `ip`
    // instead guarantees the opposite.
    uint64_t rng = init_seed(t.seed_id, STREAM_TRACKING);
    Position r_emit = t.r + t.u * (prn(&rng) * t.distance);

    // The reference side's nuclide list, taken with the reference material's
    // own cross sections loaded.
    double st_r, sa_r, snf_r;
    material_macro(s, t, t.material, st_r, sa_r, snf_r);
    list_nuclides(s, t, t.material, ref_nuc);

    // ---- the substituted material -----------------------------------
    double st_p, sa_p, snf_p;
    material_macro(s, t, mat_p, st_p, sa_p, snf_p);
    list_nuclides(s, t, mat_p, pert_nuc);
    s.material() = mat_p;

    vector<char> matched(ref_nuc.size(), 0);
    for (const auto& e : pert_nuc) {
      double d_density = e.density;
      for (size_t j = 0; j < ref_nuc.size(); ++j) {
        if (ref_nuc[j].i_nuclide == e.i_nuclide &&
            ref_nuc[j].index_sab == e.index_sab &&
            ref_nuc[j].sab_frac == e.sab_frac) {
          d_density -= ref_nuc[j].density;
          matched[j] = 1;
          break;
        }
      }
      emit_nuclide_source(
        s, t, e, d_density, pert, r_emit, base, t.seed_id, out);
    }

    // ---- whatever the reference had and the substitution does not ----
    bool any_unmatched = false;
    for (size_t j = 0; j < ref_nuc.size(); ++j)
      any_unmatched = any_unmatched || !matched[j];
    if (any_unmatched) {
      material_macro(s, t, t.material, st_r, sa_r, snf_r);
      s.material() = t.material;
      for (size_t j = 0; j < ref_nuc.size(); ++j) {
        if (!matched[j]) {
          emit_nuclide_source(s, t, ref_nuc[j], -ref_nuc[j].density, pert,
            r_emit, base, t.seed_id, out);
        }
      }
    }

  }
}

} // namespace


//==============================================================================
// Setup
//==============================================================================

void init()
{
  if (perturbations.empty())
    return;

  if (settings::run_mode != RunMode::EIGENVALUE)
    fatal_error("<local_perturbation> requires an eigenvalue calculation.");

  if (settings::survival_biasing && !settings::survival_normalization) {
    // Not an error, and not silently ignored either: the user asked for the
    // absolute form and will not get it inside a shadow tree, because there
    // it would roulette the perturbation's entire population away (see
    // apply_russian_roulette). The driver still gets exactly what was asked
    // for.
    warning("<survival_normalization> is off, but a perturbation's shadow "
            "trees will use the normalized weight cutoff regardless: their "
            "population can sit decades below unit weight, where an absolute "
            "cutoff rouletters all of it. The driver is unaffected.");
  }
  if (settings::event_based)
    fatal_error("<local_perturbation> requires history-based transport.");

  for (const auto& t : model::tallies) {
    if (t->adjoint_) {
      // Checked here, not in the Tally constructor: tallies.xml is read
      // before perturbations.xml, so BEP is not yet known to be on while
      // tallies are being built and the guard there can never fire.
      fatal_error("<local_perturbation> cannot be combined with adjoint "
                  "(super-history) tallies: both drive the revival loop, with "
                  "different generation counts and different "
                  "global-contribution rules. Run them separately.");
    }
  }
  if (settings::ifp_on) {
    // ifp() indexes simulation::ifp_source_* by current_work(), which a
    // shadow particle does not own.
    fatal_error("<local_perturbation> cannot be combined with IFP kinetics "
                "parameters. Run them separately.");
  }

  std::unordered_set<int32_t> seen_ids;
  for (const auto& p : perturbations) {
    if (!seen_ids.insert(p.id).second)
      fatal_error(fmt::format("Duplicate <local_perturbation> id {}.", p.id));
    if (p.subs.empty()) {
      fatal_error(fmt::format(
        "<local_perturbation> {} has no substitutions.", p.id));
    }
  }

  cell_perts.assign(model::cells.size(), {});
  tree_pert.clear();
  tree_class.clear();

  // Pass 1: resolve every substitution.
  for (size_t ip = 0; ip < perturbations.size(); ++ip) {
    Perturbation& p = perturbations[ip];
    std::unordered_set<int32_t> seen_cells;

    for (auto& s : p.subs) {
      auto c = model::cell_map.find(s.cell_id);
      if (c == model::cell_map.end()) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} not found.", p.id, s.cell_id));
      }
      s.cell_index = c->second;

      if (!seen_cells.insert(s.cell_index).second) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} appears twice; a perturbation "
          "may substitute each cell at most once.",
          p.id, s.cell_id));
      }

      const Cell& cell {*model::cells[s.cell_index]};
      if (cell.type_ != Fill::MATERIAL) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} must be filled with a material; "
          "the swap replaces a material, not a universe or lattice.",
          p.id, s.cell_id));
      }
      if (cell.material_.size() != 1) {
        fatal_error(fmt::format(
          "<local_perturbation> {}: cell {} must have exactly one material "
          "(no distribcell material list); the swap is per-cell.",
          p.id, s.cell_id));
      }

      if (s.mat_id == 0) {
        // Material id 0 means void, matching how an unfilled cell is stored.
        // Perturbing a sample to void is the natural "sample against nothing"
        // reference, so it has to be expressible.
        s.mat_index = MATERIAL_VOID;
      } else {
        auto m = model::material_map.find(s.mat_id);
        if (m == model::material_map.end()) {
          fatal_error(fmt::format("<local_perturbation> {}: material {} not "
                                  "found.",
            p.id, s.mat_id));
        }
        s.mat_index = m->second;
      }

      cell_perts[s.cell_index].push_back(static_cast<int>(ip));
      p.cells.push_back(s.cell_index);
    }

    // A perturbation that changes nothing is the null test: legal, and the
    // first thing to run, so warn rather than abort.
    bool null_test = true;
    for (const auto& s : p.subs) {
      if (model::cells[s.cell_index]->material_[0] != s.mat_index) {
        null_test = false;
        break;
      }
    }
    if (null_test) {
      warning(fmt::format("<local_perturbation> {} substitutes materials that "
                          "are already in place. This is the null test; rho "
                          "must come out exactly zero.",
        p.id));
    }
  }

  // Multigroup only builds macroscopic data for materials used in a cell.
  // add_substitution_temperatures() is supposed to have covered every
  // substitution target; if one slipped through, say so here rather than
  // reading a null tensor inside Mgxs::calculate_xs().
  //
  // AFTER pass 1, not before: pass 1 is what fills in mat_index. Checking
  // earlier sees -1 everywhere and silently passes.
  if (!settings::run_CE) {
    for (const auto& pert : perturbations) {
      for (const auto& sub : pert.subs) {
        if (sub.mat_index == MATERIAL_VOID || sub.mat_index < 0)
          continue;
        if (!data::mg.macro_xs_[sub.mat_index].exists_in_model) {
          fatal_error(fmt::format(
            "<local_perturbation> {} substitutes material {}, which has no "
            "multigroup data. Multigroup builds data only for materials "
            "appearing in a cell, plus perturbation targets collected by "
            "bep::add_substitution_temperatures() before the cross sections "
            "are finalized.",
            pert.id, sub.mat_id));
        }
      }
    }
  }

  // Pass 2: five source-rooted trees per perturbation, in class order. All
  // five map back to the same perturbation, because all five are transported
  // in ITS perturbed physics -- the denominator included. That is the point:
  // numerator and denominator then grow at the same rate k'^d and the level
  // is flat in depth.
  for (size_t ip = 0; ip < perturbations.size(); ++ip) {
    Perturbation& p = perturbations[ip];
    int base = static_cast<int>(tree_pert.size());
    for (int c = 0; c < N_TREE_CLASS; ++c) {
      tree_pert.push_back(static_cast<int>(ip));
      tree_class.push_back(c);
    }
    p.tree_d = base + TREE_D;
    p.tree_fp = base + TREE_FP;
    p.tree_fn = base + TREE_FN;
    p.tree_lp = base + TREE_LP;
    p.tree_ln = base + TREE_LN;
  }

  // NOTE: settings::super_n_generation is deliberately NOT touched here.
  // The driver reads it in event_check_limit_and_revive(), so changing it
  // would alter driver behaviour -- and BEP must leave the driver
  // bit-identical to a stock run. Shadow particles get their own revival
  // limit from bep_n_generation, keyed off bep_tree(). Same reasoning for
  // simulation::superhistory_on, which BEP no longer forces on:
  // create_fission_sites() tests simulation::bep_on directly instead.

  int nd = settings::bep_n_generation + 1;
  size_t np = perturbations.size();
  tau.assign(tree_pert.size() * nd, 0.0);
  batch_tau.assign(tree_pert.size() * nd, 0.0);
  pooled_tau.assign(tree_pert.size() * nd, 0.0);
  ell_sum.assign(np * nd, 0.0);
  ell_cross.assign(np * np * nd, 0.0);
  n_active_batches = 0;
  // Unit weight until a generation has been run to measure from, which is
  // exactly what an eigenvalue calculation does anyway.
  tree_site_weight.assign(tree_pert.size(), 1.0);
  tree_weight_scale.assign(tree_pert.size(), 1.0);
  tree_sources.assign(tree_pert.size(), 0);
  batch_sources.assign(tree_pert.size(), 0);
  batch_root_sq.assign(tree_pert.size(), 0.0);
  batch_pair_sq.assign(perturbations.size() * 2, 0.0);
  batch_pair_n.assign(perturbations.size() * 2, 0);
  thread_root_sq.assign(
    static_cast<size_t>(num_threads()) * tree_pert.size(), 0.0);
  thread_tau.assign(static_cast<size_t>(num_threads()) * tau_stride(), 0.0);
  thread_root_weight.assign(num_threads(), 0.0);
  thread_tracks.assign(num_threads(), {});
  thread_source_roots.assign(num_threads(), {});
  tracks.clear();
  source_roots.clear();

  // Everything the run records is now O(n_pert * L) and independent of its
  // length, so there is no allocation to warn about.

  size_t n_cells_touched = 0;
  for (const auto& cp : cell_perts)
    if (!cp.empty())
      ++n_cells_touched;
  write_message(
    fmt::format("BEP: {} perturbation(s) over {} cell(s), {} shadow trees, "
                "L = {}.",
      np, n_cells_touched, tree_pert.size(), settings::bep_n_generation),
    5);
}

void reset_generation()
{
  // Decide once per generation whether BEP does anything at all. Shadow trees
  // are pure overhead before the fission source has converged, and leaving
  // the flag clear keeps the branch test out of the transport hot path
  // entirely during inactive batches rather than testing and discarding on
  // every event.
  // Decide once per generation whether BEP does anything at all, exactly as
  // setup_active_tallies() decides superhistory_on. Shadow trees are pure
  // overhead before the fission source has converged, and leaving the flag
  // clear keeps the branch test out of the transport hot path entirely
  // during inactive batches rather than testing and discarding on every
  // event.
  simulation::bep_on =
    !perturbations.empty() && simulation::current_batch > settings::n_inactive;
  if (!simulation::bep_on)
    return;
  std::fill(tau.begin(), tau.end(), 0.0);
  std::fill(thread_tau.begin(), thread_tau.end(), 0.0);
  std::fill(tree_sources.begin(), tree_sources.end(), 0);
  std::fill(thread_root_sq.begin(), thread_root_sq.end(), 0.0);
  for (auto& v : thread_tracks)
    v.clear();
  for (auto& v : thread_source_roots)
    v.clear();
  tracks.clear();
  source_roots.clear();
}

//==============================================================================
// Branch detection (driver side)
//==============================================================================

void add_substitution_temperatures(vector<vector<double>>& kTs)
{
  if (perturbations.empty())
    return;

  // Resolve the ids here rather than reading Substitution::mat_index.
  // init() is what fills those in, and it does not run until simulation
  // setup -- long after the cross sections are finalized. Reading them at
  // this point would silently see -1 for every substitution and add no
  // temperatures at all, which is exactly the bug this function exists to
  // prevent. model::cell_map and model::material_map are both populated by
  // now, so look up by id.
  for (const auto& pert : perturbations) {
    for (const auto& sub : pert.subs) {
      if (sub.mat_id == 0)
        continue; // void: nothing to build data for

      auto m = model::material_map.find(sub.mat_id);
      auto c = model::cell_map.find(sub.cell_id);
      // Bad ids are not diagnosed here -- init() reports them properly, with
      // the perturbation id attached. Just skip.
      if (m == model::material_map.end() || c == model::cell_map.end())
        continue;

      // The substituted material is evaluated at the HOST cell's
      // temperature, so those are exactly the temperatures it needs.
      for (double sqrtkT : model::cells[c->second]->sqrtkT_) {
        double kT = sqrtkT * sqrtkT;
        if (!contains(kTs[m->second], kT))
          kTs[m->second].push_back(kT);
      }
    }
  }
}

void add_substitution_nuclide_temperatures(vector<vector<double>>& nuc_temps)
{
  if (perturbations.empty())
    return;

  for (const auto& pert : perturbations) {
    for (const auto& sub : pert.subs) {
      if (sub.mat_id == 0)
        continue;

      auto m = model::material_map.find(sub.mat_id);
      auto c = model::cell_map.find(sub.cell_id);
      if (m == model::material_map.end() || c == model::cell_map.end())
        continue;

      const auto& mat = model::materials[m->second];
      for (double sqrtkT : model::cells[c->second]->sqrtkT_) {
        // Kelvin here, unlike add_substitution_temperatures().
        double temperature = sqrtkT * sqrtkT / K_BOLTZMANN;
        for (int i_nuc : mat->nuclide_) {
          if (!contains(nuc_temps[i_nuc], temperature))
            nuc_temps[i_nuc].push_back(temperature);
        }
      }
    }
  }
}

void record_track(Particle& p, int32_t cell_index, double distance)
{
  // Caller has established that BEP is on this batch, that p is a trunk
  // and that cell_index is touched by at least one perturbation.
  if (!p.type().is_neutron())
    return;
  if (p.wgt() == 0.0 || distance <= 0.0)
    return;
  if (!settings::run_CE) {
    // The multigroup source would need the group-to-group transfer matrices
    // of both materials, which is the same work as the continuous-energy
    // scattering kernel deferred to stage 3. Refusing is better than a
    // silently absorption-only source.
    fatal_error("<local_perturbation> currently requires continuous-energy "
                "transport: the perturbation source is not implemented for "
                "multigroup data.");
  }

  TrackSite site;
  site.r = p.r();
  site.u = p.u();
  site.E = p.E();
  site.wgt = p.wgt();
  site.time = p.time();
  site.distance = distance;
  // The reference cross sections come for free: the driver has just computed
  // them for this very flight, so the source costs no cross-section lookup
  // at all on the reference side.
  site.sigma_t = p.macro_xs().total;
  site.sigma_a = p.macro_xs().absorption;
  site.nu_sigma_f = p.macro_xs().nu_fission;
  site.sqrtkT = p.sqrtkT();
  site.density_mult = p.density_mult();
  site.cell = cell_index;
  site.material = p.material();
  // Seeded on the identity of the segment, not its arrival order, so the
  // shadow trees do not depend on which thread got here first. (id,
  // n_tracks, n_event) is unique for a driver particle within a generation
  // -- n_event alone is not, because event_revive_from_secondary() resets
  // it -- and total_gen distinguishes generations.
  site.seed_id = combine_ids(
    {simulation::total_gen, p.id(), p.n_tracks(), p.n_event()});

  // No synchronisation: this runs in the transport loop, and an omp critical
  // here serialises every thread on a push_back. The vectors are merged in
  // run_shadow_pass(), where the order no longer matters because each site
  // carries its own seed.
  thread_tracks[thread_num()].push_back(site);

  // The driver is deliberately NOT stopped, NOT tagged and NOT sampled from:
  // it carries on in the reference state exactly as in a stock run, so k, the
  // fission source and all ordinary tallies are untouched. Every segment of
  // every history through a touched cell is recorded, which is exactly the
  // track-length estimator of the perturbation source integral.
}

double root_weight()
{
  return thread_root_weight.empty() ? 0.0 : thread_root_weight[thread_num()];
}

void score_site(int tree, int super_gen, double wgt)
{
  int depth = super_gen - 1; // shadow roots live at super_gen == 1
  if (depth < 0 || depth > settings::bep_n_generation)
    return;
  // The innermost operation of the whole feature, once per shadow fission
  // site. Writing to a per-thread slab rather than an atomic add on shared
  // memory: the shared array is a few hundred bytes, so every thread would
  // contend for the same handful of cache lines billions of times.
  thread_tau[static_cast<size_t>(thread_num()) * tau_stride() +
             tau_index(tree, depth)] += wgt;
}

//==============================================================================
// Shadow pass
//==============================================================================

//! Sample this generation's DENOMINATOR roots from the fission bank.
//!
//! The bank is the reference fission source already divided by the eigenvalue
//! -- create_fission_sites() banks an expected w * nu Sigma_f / (Sigma_t *
//! keff) sites per collision -- so it is a realisation of (1/k) F psi, which
//! is exactly the denominator the level needs and exactly the normalisation
//! level()'s kk assumes.
//!
//! Sampling rather than taking the whole bank: each site is kept with
//! probability p and carries 1/p, which is unbiased and makes the cost of the
//! denominator a tunable knob instead of a full extra transport of the
//! problem at every depth. The draws come from a stream seeded on the
//! generation and the rank alone, so the sample does not depend on thread
//! timing or on the order the bank was filled in.
void sample_denominator_roots()
{
  auto n_bank = static_cast<int64_t>(simulation::fission_bank.size());
  if (n_bank == 0)
    return;

  // The root count is TIED to the denominator's site weight; it is not a
  // knob of its own.
  //
  // It used to be one, with an optimum of its own derived in section 4b. That
  // was a mistake of a particular kind: the roots and the sites they grow are
  // the same population sampled at two different weights, and letting the two
  // be chosen independently let them disagree. They did, badly. Measured on a
  // tuned run, the denominator carried about 8000 roots to yield 78 sites --
  // 95% of the whole denominator cost went into roots that were immediately
  // rouletted away by a site weight of 100.
  //
  // Keeping a site with probability 1/w_D makes the kept roots carry weight
  // w_D, which is the weight their descendants will be banked at anyway. The
  // denominator then costs about L*N_D instead of M_D + (L-1)*N_D, and the
  // variance per site changes by roughly a tenth. One knob, and the roots
  // cannot disagree with the sites about how finely the denominator is
  // sampled.
  //
  // The finest denominator across perturbations sets it, since a kept bank
  // site is rooted once in EACH of them and a shared sample must not starve
  // the one that wants the most roots.
  double w_fine = 0.0;
  for (const auto& p : perturbations) {
    double w = site_weight(p.tree_d);
    if (w > 0.0 && (w_fine == 0.0 || w < w_fine))
      w_fine = w;
  }
  double p_keep = (w_fine > 1.0) ? 1.0 / w_fine : 1.0;
  double scale = 1.0 / p_keep;

  for (int64_t i = 0; i < n_bank; ++i) {
    const SourceSite& s = simulation::fission_bank[i];

    // Keyed on the SITE, never on its index in the bank. The bank is filled
    // by thread_safe_append() during transport, so a site's index depends on
    // which thread got there first and differs between two runs of the same
    // seed. Driving either the sampling or the tree's seed from it made the
    // whole denominator irreproducible -- two identical runs differed by
    // over a per cent. (parent_id, progeny_id) is set deterministically when
    // the site is created and identifies it uniquely within the generation.
    int64_t sid =
      combine_ids({simulation::total_gen, s.parent_id, s.progeny_id});

    // The keep/reject draw comes from a DIFFERENT stream than the one the
    // kept site's tree is grown on, for the same reason the sampled source
    // roots do (see the note in emit_nuclide_source): a walk must not replay
    // the numbers that decided its own existence.
    //
    // This one was the worse of the two. The test keeps a site when its first
    // tracking number is below p_keep, and the tree was then started from
    // that same number -- so every surviving root began with a uniform
    // conditioned to [0, p_keep), and its first flight, -ln(xi)/Sigma_t, was
    // never shorter than ln(1/p_keep) mean free paths. At the usual
    // n_roots/bank ratio of about a tenth that is 2.3 mfp, on EVERY
    // denominator root, in every case -- biasing D, which sits under every
    // level this estimator reports.
    if (p_keep < 1.0) {
      uint64_t rs = init_seed(combine_ids({sid, 1}), STREAM_TRACKING);
      if (prn(&rs) >= p_keep)
        continue;
    }

    for (const auto& pert : perturbations) {
      SourceRoot root;
      root.r = s.r;
      root.u = s.u;
      root.E = s.E;
      root.wgt = s.wgt * scale;
      root.time = s.time;
      root.tree = pert.tree_d;
      // Each perturbation grows this site in ITS OWN perturbed physics, but
      // from the same seed, so the denominators of two perturbations differ
      // only where their physics does.
      root.seed_id = sid;
      source_roots.push_back(root);
    }
  }
}

void run_shadow_pass()
{
  if (!simulation::bep_on)
    return;

  // The eigenvalue the bank was normalised by, and hence the kk of level().
  keff_norm = simulation::keff;

  // Merge the per-thread track vectors and put them in a deterministic
  // order. The order the threads recorded them in varies run to run; sorting
  // by the seed, which is a property of the segment itself, makes the whole
  // shadow pass -- including the order the per-thread tau slabs are summed in
  // -- reproducible for a given thread count.
  tracks.clear();
  for (const auto& v : thread_tracks)
    tracks.insert(tracks.end(), v.begin(), v.end());
  std::stable_sort(tracks.begin(), tracks.end(),
    [](const TrackSite& a, const TrackSite& b) {
      return a.seed_id < b.seed_id;
    });

  // ---- 1. the perturbation source, dH psi -------------------------------
  for (auto& v : thread_source_roots)
    v.clear();
  auto n_tracks = static_cast<int64_t>(tracks.size());
#pragma omp parallel
  {
    // One scratch particle per thread, reused across segments: it exists only
    // to hold the cross sections of a material the driver is not in, and to
    // sample a fission neutron from it.
    Particle scratch;
    scratch.type() = ParticleType::neutron();
    vector<SourceRoot>& out = thread_source_roots[thread_num()];
#pragma omp for schedule(static)
    for (int64_t i = 0; i < n_tracks; ++i)
      emit_perturbation_source(tracks[i], scratch, out);
  }

  source_roots.clear();
  for (const auto& v : thread_source_roots)
    source_roots.insert(source_roots.end(), v.begin(), v.end());

  // ---- 2. the denominator, a sample of the reference fission source -----
  sample_denominator_roots();

  if (source_roots.empty())
    return;

  // One deterministic order for the whole root list, whatever order the
  // threads and the bank produced it in.
  // STABLE, and the key is deliberately not unique: the two members of a
  // +/- pair share a seed, and an (n,xn) source contributes several roots
  // with the same seed AND the same tree. std::sort would order those
  // arbitrarily, the static schedule below would then hand them to different
  // threads, and the per-thread tau sums would stop being reproducible. The
  // input order is already deterministic -- per-thread vectors merged in
  // thread order, each filled by a static schedule over the sorted tracks --
  // so a stable sort keeps it that way.
  std::stable_sort(source_roots.begin(), source_roots.end(),
    [](const SourceRoot& a, const SourceRoot& b) {
      if (a.seed_id != b.seed_id)
        return a.seed_id < b.seed_id;
      return a.tree < b.tree;
    });

  // ---- 3. grow every root L generations in its perturbation's physics ---
  //
  // Static, not dynamic: with the roots in a fixed order it gives each thread
  // a fixed chunk, so the per-thread partial sums are reproducible. Tree cost
  // varies a lot -- it is a branching process -- but with many roots per
  // thread that averages out. Switch to dynamic if load imbalance ever shows
  // up, at the cost of bit-reproducibility.
  auto n = static_cast<int64_t>(source_roots.size());
  root_x.assign(source_roots.size(), 0.0);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; ++i) {
    run_one_tree(source_roots[i], &root_x[i]);
  }

  // ---- 4. the SIGNED within-batch spread of each numerator channel --------
  //
  // This is what retires `r`.
  //
  // The tuning rule needs the variance of the signed numerator, N_L = L+ - L-
  // and N_F = F+ - F-, not the variance of L+ and L- separately. Those two are
  // strongly correlated -- they are grown from the same source event on common
  // random numbers, which is the whole point of sharing a seed -- so the
  // variance of the difference is smaller than the sum of the variances by a
  // factor (1 - r), and r is neither known nor safely estimable from batch
  // statistics (section 6).
  //
  // The derivation's answer was to set r = 0 and call it conservative. It is
  // not conservative here: it makes the modelled splittable variance EXCEED
  // the measured total, so the floor comes out negative and the rule refuses
  // to act at all. Measured at unit weights on the test problem, v_meas = 1.32
  // against a modelled G_L*rv_L = 1.40.
  //
  // Forming the signed combination per common-random-number group and squaring
  // THAT puts (1 - r) inside the measurement, where it needs no coefficient
  // and no assumption. A group is a maximal run of roots sharing a seed_id,
  // which the stable sort above has already made adjacent; each group belongs
  // to exactly one channel, since the F root, the absorption root and the
  // scattered pair are drawn on different keys.
  if (mpi::master) {
    size_t i = 0;
    while (i < source_roots.size()) {
      const int64_t sid = source_roots[i].seed_id;
      size_t j = i;
      double acc[2] {0.0, 0.0}; // 0 = L channel, 1 = F channel
      bool seen[2] {false, false};
      while (j < source_roots.size() && source_roots[j].seed_id == sid) {
        const int tr = source_roots[j].tree;
        const int cls = tree_class[tr];
        int ch = -1;
        double sign = 0.0;
        switch (cls) {
        case TREE_LP: ch = 0; sign = +1.0; break;
        case TREE_LN: ch = 0; sign = -1.0; break;
        case TREE_FP: ch = 1; sign = +1.0; break;
        case TREE_FN: ch = 1; sign = -1.0; break;
        default: break; // TREE_D is the denominator, measured separately
        }
        if (ch >= 0) {
          acc[ch] += sign * root_x[j];
          seen[ch] = true;
        }
        ++j;
      }
      for (int ch = 0; ch < 2; ++ch) {
        if (!seen[ch])
          continue;
        // Which perturbation this group belongs to: every root in a group
        // feeds one perturbation's trees.
        int ip = -1;
        for (size_t q = i; q < j; ++q) {
          const int cls = tree_class[source_roots[q].tree];
          if ((ch == 0 && (cls == TREE_LP || cls == TREE_LN)) ||
              (ch == 1 && (cls == TREE_FP || cls == TREE_FN))) {
            ip = tree_pert[source_roots[q].tree];
            break;
          }
        }
        if (ip < 0)
          continue;
        const size_t slot = static_cast<size_t>(ip) * 2 + ch;
        batch_pair_sq[slot] += acc[ch] * acc[ch];
        batch_pair_n[slot] += 1;
      }
      i = j;
    }
  }
}

void update_site_weights()
{
  int nd = settings::bep_n_generation + 1;

  // Total weight each population carried this generation, summed over depth.
  vector<double> total(tree_pert.size(), 0.0);
  for (size_t t = 0; t < tree_pert.size(); ++t) {
    for (int d = 0; d < nd; ++d)
      total[t] += tau[tau_index(static_cast<int>(t), d)];
  }

  // Only weight_scale is set here, per generation. It describes the
  // population rather than a choice about it -- what a typical particle in it
  // weighs relative to the denominator's, which for a weak perturbation is
  // several decades below one -- and it is measured whatever the splitting
  // flag says, because it is what a weight cutoff inside the tree has to be
  // judged against: against the denominator's scale the whole population sits
  // below the cutoff and survival biasing would roulette all of it away.
  //
  // The SITE WEIGHT is chosen at the batch boundary instead, in
  // choose_site_weights(), because the quantity it needs -- the measured
  // relative variance of the worth -- is a batch statistic, and because a
  // weight held constant for a whole batch makes the banked-site count it
  // solves for exact rather than approximate.
  for (const auto& p : perturbations) {
    double w_d = total[p.tree_d];
    if (w_d <= 0.0)
      continue; // nothing measured yet

    const int pairs[2][2] {{p.tree_fp, p.tree_fn}, {p.tree_lp, p.tree_ln}};
    for (const auto& pair : pairs) {
      double w = total[pair[0]] + total[pair[1]];
      if (w <= 0.0)
        continue;
      // The two members of a +/- pair are one population as far as any of
      // this is concerned: they exist to be subtracted from each other, and
      // sampling them differently would decorrelate exactly the pair whose
      // correlation the whole scheme rests on.
      double q = std::pow(10.0, std::floor(std::log10(w / w_d) + 0.5));
      q = std::min(1.0, std::max(MIN_SITE_WEIGHT, q));
      tree_weight_scale[pair[0]] = q;
      tree_weight_scale[pair[1]] = q;
    }
  }
}

void accumulate_generation()
{
  if (!simulation::bep_on)
    return;

  // Sum the per-thread slabs, in thread order.
  std::fill(tau.begin(), tau.end(), 0.0);
  int stride = tau_stride();
  for (int t = 0; t < num_threads(); ++t) {
    for (int i = 0; i < stride; ++i)
      tau[i] += thread_tau[static_cast<size_t>(t) * stride + i];
  }

  // The per-root spread, summed into the batch. Rank-local and taken before
  // the reduction below, exactly as the site weights are: these feed a weight
  // window, not an estimator, so a rank measuring from its own trees costs
  // nothing in correctness. (With MPI the counts are therefore per-rank; the
  // site-weight rule has always been.)
  for (size_t t = 0; t < tree_pert.size(); ++t) {
    batch_sources[t] += tree_sources[t];
    for (int th = 0; th < num_threads(); ++th)
      batch_root_sq[t] +=
        thread_root_sq[static_cast<size_t>(th) * tree_pert.size() + t];
  }

  // Set the next generation's site weights from this generation's tau, before
  // the reduction below: every rank then measures from its own trees. The
  // value is a weight window and not an estimator, so a rank choosing its own
  // costs nothing in correctness, and rounding to a power of ten means ranks
  // agree on it in practice anyway.
  update_site_weights();

#ifdef OPENMC_MPI
  if (mpi::n_procs > 1) {
    // Sum tau across ranks before recording: the estimator is a ratio of sums
    // over every progenitor, not a mean of per-rank ratios.
    //
    // One unconditional call with a separate receive buffer, as everywhere
    // else in the code -- MPI_Reduce ignores recvbuf off the root, so there
    // is nothing to branch on. tau is a few kB, and the MPI_IN_PLACE form
    // that avoids the temporary is only worth its extra branch where a
    // master-only block already exists for other reasons (see the tally
    // write in state_point.cpp).
    //
    // Reduce rather than allreduce: only the master writes the statepoint,
    // so no other rank reads tau again.
    vector<double> reduced(tau.size(), 0.0);
    mpi::reduce<double>(
      tau.data(), reduced.data(), tau.size(), MPI_SUM, 0, mpi::intracomm);
    if (mpi::master)
      tau.swap(reduced);
  }
#endif

  // Accumulate into the batch, and form no ratio here. A shadow tree is a
  // branching process that can go extinct, so a single generation's tau can
  // be zero; summing over every root of the whole batch before dividing is
  // what makes this robust to that, exactly as ordinary IFP estimators are.
  // Dropping degenerate generations instead would bias the worth, since
  // extinction correlates with the strength of the perturbation.
  //
  // Only the master holds the global sum after the reduction, so only the
  // master keeps the record. n_generations counts on every rank so the two
  // stay in step if a later change ever wants the totals elsewhere.
  if (mpi::master) {
    for (size_t i = 0; i < tau.size(); ++i) {
      batch_tau[i] += tau[i];
      pooled_tau[i] += tau[i];
    }
  }
  ++n_generations;
}

//! Choose each numerator population's site weight, from the batch just
//! closed. See docs/bep_autotune.md for the derivation; the short form is
//!
//!     N_t* = sqrt( G_t * C0 / B )
//!
//! where N_t is the population's banked sites per batch, G_t is a
//! dimensionless number saying how hard that population's sampling noise
//! pushes on the worth, C0 is the cost of everything that is not a tunable
//! population, and B is the irreducible floor of the worth's relative
//! variance. The scale closes exactly -- at the optimum V/C = B/C0 -- which
//! is what removes the run's total cost and total variance from the answer
//! and makes this a constant map rather than an iteration needing a
//! contraction guard.
//!
//! G_t is where the cancellation enters, and it is the whole reason the old
//! rule (match the denominator's population) was wrong:
//!
//!     G_D = 1/k^2
//!     G_L = [ (L+ + L-) / S ]^2
//!     G_F = [ (F+ + F-) / (k S) ]^2      S = N_F/k + N_L, the numerator
//!
//! The bracket is the amplification a signed source suffers: the noise is set
//! by the populations, the answer by their difference. Measured 8 for a
//! strong absorber and 65 for a dissolved-U235 case, so those populations
//! want 17x and 143x the denominator's sites -- not the 1x the old rule gave
//! them.
//!
//! Correlation within a pair is taken as ZERO, i.e. G_t uses (L+ + L-) rather
//! than (L+ + L-)(1 - r). That is the pessimistic bound and it over-samples;
//! the optimum is flat enough (a factor of three costs at most a third of the
//! figure of merit) that measuring r is not worth the accumulators.
namespace {

void choose_site_weights()
{
  // The off switch, and it is checked FIRST -- before the diagnostic
  // override, not after it.
  //
  // It used to be the other way round, so a run with
  // perturbation_site_splitting = false and a perturbation_site_weight set
  // still had its numerator weights moved off 1. A switch a second setting
  // can quietly overrule is not an off switch. Off now means off: every site
  // weight stays at 1, and sample_denominator_roots() then keeps the whole
  // fission bank -- i.e. exactly what an ordinary eigenvalue calculation
  // banks, and the state this feature had before any tuning existed.
  if (!settings::perturbation_site_splitting)
    return;

  if (settings::bep_site_weight > 0.0) {
    // An explicit override short-circuits the rule. Used to sweep the site
    // weight and measure the figure of merit against it, which is the only
    // way to confirm the optimum is where the derivation says it is.
    for (const auto& p : perturbations) {
      for (int tree : {p.tree_fp, p.tree_fn, p.tree_lp, p.tree_ln})
        tree_site_weight[tree] = settings::bep_site_weight;
    }
    return;
  }


  // Five active batches before anything measured here is trusted at all.
  if (n_active_batches < 5)
    return;

  int nd = settings::bep_n_generation + 1;
  int L = settings::bep_n_generation;
  size_t np = perturbations.size();

  // ---- the cost model, checked against the history counter -----------------
  //
  // Cost is HISTORIES, and a root is a history: run_one_tree() transports one
  // per root (bep.cpp, transport_history_based_single_particle) and one per
  // REVIVED site. The revival gate is `super_gen < gen_limit` with
  // gen_limit = L + 1, and the sites banked at depth L carry super_gen = L+1,
  // so they are scored and never revived. A tree therefore costs
  //
  //     M_t  +  sum_{d=1..L-1} N_t(d)   ~   M_t + (L-1) * N_t
  //
  // and the unit cost of a site-count knob is kappa = L - 1, not L. This was
  // L, and it is not a matter of taste: reconstructing the per-generation
  // history budget from tau_pooled and n_sources in a finished statepoint
  // matches the measured n_histories to 1.5% with L-1 and misses by 12-14%
  // with L, on two independently tuned arms (see docs/bep_autotune.md 11).
  // Per-knob unit cost. A tree costs M_t + sum_{d=1..L-1} N_t(d): the
  // depth-L sites are banked and scored but never revived (the gate is
  // super_gen < L+1 and they carry L+1), which the history counter confirms
  // to 1.5% against 12-14% for L. So a numerator knob costs L-1 per site.
  //
  // The DENOMINATOR costs L, not L-1, because its root count is tied to its
  // site weight: sample_denominator_roots() keeps a bank site with
  // probability 1/w_D, so splitting the denominator finer buys more roots as
  // well as more sites, and pays for both.
  const double kappa_num = static_cast<double>(L - 1);
  const double kappa_den = static_cast<double>(L);
  if (kappa_num <= 0.0)
    return;

  // The UNTUNABLE cost: the driver, plus the numerator roots. The numerator
  // roots are emitted one per driver track segment per changed nuclide, so
  // they are set by the driver and are not a knob. The denominator's roots
  // are deliberately NOT in here any more -- they are tied to w_D, so they
  // are part of what the denominator knob pays for, and putting a tuned cost
  // into a fixed one double-counts the knob being chosen.
  const double gpb = static_cast<double>(settings::gen_per_batch);
  double c_untuned = static_cast<double>(settings::n_particles) * gpb;
  for (size_t t = 0; t < tree_sources.size(); ++t) {
    if (tree_class[t] != TREE_D)
      c_untuned += static_cast<double>(tree_sources[t]) * gpb;
  }
  if (c_untuned <= 0.0)
    return;

  // The per-batch relative variance of tree t's depth-L weight, measured
  // across that tree's own roots:
  //
  //     rv_t = sum(x^2)/(sum x)^2 - 1/M_t      (x = one root's depth-L weight)
  //
  // Note what this is and is not. It is the FULL relative variance, including
  // every layer of banking discreteness between depth 1 and depth L -- not a
  // separate "root" term sitting beside a 1/N_t the model supplies. A
  // branching process banked at weight w accumulates one layer of
  // banking-plus-transport variance per generation, each of them proportional
  // to w, so relvar(X_t) ~ a_t/N_t with a_t = 1 + (L-1)*sigma_Z^2, which is
  // 5 to 20 here and not the <= 1 the derivation assumed. Splitting controls
  // ALL of it.
  //
  // So rv_t is not something to add to 1/N_t -- that double-counts -- and it
  // is not weight-independent either. It is the measurement that replaces the
  // modelled coefficient outright.
  auto root_relvar = [&](int tree) -> double {
    double m = static_cast<double>(batch_sources[tree]);
    double sx = batch_tau[tau_index(tree, L)];
    double sxx = batch_root_sq[tree];
    if (m <= 1.0 || sx <= 0.0 || sxx <= 0.0)
      return 0.0;
    double rv = sxx / (sx * sx) - 1.0 / m;
    return rv > 0.0 ? rv : 0.0;
  };

  // ---- pass 1: the measured slope of each knob, and the measured floor ----
  //
  // The closed form is back, because eq (4) on measured V and counted C does
  // NOT avoid needing the floor. With N_t = sqrt(g_t)*s and Gamma = sum
  // sqrt(g_t), C(s) = C_0 + kappa*Gamma*s and V(s) = B + Gamma/s, so the fixed
  // point of s <- sqrt(C/(kappa V)) is
  //
  //     kappa*B*s^2 + kappa*Gamma*s = C_0 + kappa*Gamma*s   ->   kappa*B*s^2 = C_0
  //
  // Gamma cancels identically and s* = sqrt(C_0/(kappa*B)) either way. The
  // iterated form merely finds B implicitly, by splitting until V has fallen
  // to the floor -- which, where the floor is small, means splitting until
  // cost dominates. Measured: it drove w_L to 1e-3 and the work to 14x the
  // feature switched off.
  //
  // So B is faced directly, and it is SUBTRACTED FROM A MEASUREMENT RATHER
  // THAN FROM A MODEL:
  //
  //     B = v_meas - [ relvar_within(S) + relvar_within(D) ]
  //
  // with both bracketed terms measured within the batch, across independent
  // source events. Nothing in it is a coefficient.
  struct PertPlan {
    double g_d, g_l, g_f;   // measured gain of each knob
    double p_d, p_l, p_f;   // carried weights
    double w_d;             // largest denominator weight the expansion allows
    bool ok {false};
  };
  vector<PertPlan> plan(np);
  double b_tot = 0.0;

  for (size_t ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];

    double d_w = batch_tau[tau_index(p.tree_d, L)];
    double fp = batch_tau[tau_index(p.tree_fp, L)];
    double fn = batch_tau[tau_index(p.tree_fn, L)];
    double lp = batch_tau[tau_index(p.tree_lp, L)];
    double ln = batch_tau[tau_index(p.tree_ln, L)];
    if (d_w <= 0.0)
      continue;

    // S from the POOLED totals, never from the batch just closed: one batch's
    // S is a difference of nearly equal populations, sign-indefinite with a
    // standard deviation of order its own mean, and dividing by it is
    // dividing by a variable whose reciprocal has no finite expectation.
    double n_b = static_cast<double>(n_active_batches);
    double pfp = pooled_tau[tau_index(p.tree_fp, L)];
    double pfn = pooled_tau[tau_index(p.tree_fn, L)];
    double plp = pooled_tau[tau_index(p.tree_lp, L)];
    double pln = pooled_tau[tau_index(p.tree_ln, L)];
    double pd = pooled_tau[tau_index(p.tree_d, L)];
    double s_num = (pfp - pfn) / keff_norm + (plp - pln);
    if (pd <= 0.0 || s_num == 0.0)
      continue;

    // The measured relative variance of the worth over active batches.
    double mean = ell_sum[ip * nd + L] / n_b;
    if (mean == 0.0)
      continue;
    double sq = ell_cross[(ip * np + ip) * nd + L];
    double var = (sq - n_b * mean * mean) / (n_b - 1.0);
    if (var <= 0.0)
      continue;
    double v_meas = var / (mean * mean);

    // The SIGNED within-batch spread of each numerator channel, measured over
    // common-random-number groups, so that (1 - r) is inside the measurement.
    // Scaled to the worth: the L channel enters S directly, the F channel
    // divided by k.
    auto chan_var = [&](int ch) -> double {
      const size_t slot = static_cast<size_t>(ip) * 2 + static_cast<size_t>(ch);
      double m = static_cast<double>(batch_pair_n[slot]);
      double sxx = batch_pair_sq[slot];
      if (m <= 1.0 || sxx <= 0.0)
        return 0.0;
      double sx = (ch == 0) ? (lp - ln) : (fp - fn);
      // Var of the channel sum, estimated from its own groups, then made
      // relative to the POOLED S (the scale the worth is actually read at).
      double v = sxx - sx * sx / m;
      if (v <= 0.0)
        return 0.0;
      // s_num is the POOLED numerator, summed over n_b batches, while v is
      // ONE batch's variance. Both sides must be per batch or the ratio is
      // wrong by n_b^2.
      double s_batch = s_num / n_b;
      double scale_to_s = (ch == 0) ? 1.0 : 1.0 / keff_norm;
      return v * scale_to_s * scale_to_s / (s_batch * s_batch);
    };
    double rvs_l = chan_var(0);
    double rvs_f = chan_var(1);
    double rv_d = root_relvar(p.tree_d);

    // The floor: what is left of the measured variance once every splittable
    // part that was MEASURED is taken out.
    double b = v_meas - (rvs_l + rvs_f + rv_d);
    if (!(b > 0.0)) {
      // Hold the last usable floor rather than freeze. A batch where the
      // subtraction overshoots says nothing about the floor; it says this
      // batch was lucky. Leaving the weights alone until it recovers is the
      // conservative reading, and unlike the r = 0 form this is an occasional
      // event rather than the usual one.
      continue;
    }

    // The gain of each knob, measured: relvar ~ g/N, so g = relvar * N.
    double n_d = d_w / site_weight(p.tree_d);
    double n_l = (lp + ln) / site_weight(p.tree_lp);
    double n_f = (fp + fn) / site_weight(p.tree_fp);

    // The delta method's domain of validity, binding on the DENOMINATOR
    // alone: rho is linear in S -- a noisy S is noisy, not biased -- but a
    // ratio in D. rv_D is proportional to w_D, so the largest weight still
    // satisfying rv_D <= EPS^2 extrapolates linearly from where the run is.
    const double eps2 = DELTA_METHOD_EPS * DELTA_METHOD_EPS;
    double w_now = site_weight(p.tree_d);
    double w_cap = (rv_d > 0.0) ? w_now * eps2 / rv_d : 0.0;

    plan[ip] = {rv_d * n_d, rvs_l * n_l, rvs_f * n_f,
      d_w, lp + ln, fp + fn, w_cap, true};
    b_tot += b;
  }

  if (b_tot <= 0.0)
    return;

  // ---- pass 2: N_t* = sqrt( g_t/kappa_t * C_0/B ), the closed form --------
  //
  // C_0 is the UNTUNABLE cost: the driver, plus the numerator roots, which
  // are set by the driver's tracks and are not a knob. The denominator's
  // roots are no longer in it, because they are no longer independent of the
  // denominator's site weight -- sample_denominator_roots() keeps a bank site
  // with probability 1/w_D, which is also why kappa_D is L and not L-1: the
  // denominator pays for its roots as it splits, the numerators do not.
  const double scale = std::sqrt(c_untuned / b_tot);

  for (size_t ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];
    const PertPlan& q = plan[ip];
    if (!q.ok)
      continue;

    const struct {
      int a, b2;
      double g, weight, cap, kap;
    } groups[3] {{p.tree_d, p.tree_d, q.g_d, q.p_d, q.w_d, kappa_den},
      {p.tree_lp, p.tree_ln, q.g_l, q.p_l, 0.0, kappa_num},
      {p.tree_fp, p.tree_fn, q.g_f, q.p_f, 0.0, kappa_num}};

    for (const auto& grp : groups) {
      if (grp.weight <= 0.0 || grp.g <= 0.0)
        continue;
      double target = std::sqrt(grp.g / grp.kap) * scale;
      if (!(target > 0.0))
        continue;

      // Rounded to the nearest HALF DECADE. Coarse enough that the value does
      // not chase sampling noise from batch to batch, fine enough to cost at
      // most 9% of the figure of merit at the worst rounding -- where whole
      // decades, which this used to use, can cost 37%.
      double w = std::pow(
        10.0, 0.5 * std::floor(2.0 * std::log10(grp.weight / target) + 0.5));

      // At most one half decade per batch. The rule reads its own past
      // output, and bounding the step keeps one noisy batch from throwing it
      // a long way. It cannot move the fixed point, only the approach.
      double w_prev = tree_site_weight[grp.a];
      if (w_prev > 0.0) {
        const double step = std::sqrt(10.0);
        w = std::min(std::max(w, w_prev / step), w_prev * step);
      }

      // The validity cap, applied to the REALISED weight, after the rounding
      // and after the damper -- a hard bound outranks a damper, and clamping
      // before the rounding did not survive it at all (a target clamped to
      // 100 sites came out at 77.6).
      if (grp.cap > 0.0 && w > grp.cap)
        w = std::pow(10.0, 0.5 * std::floor(2.0 * std::log10(grp.cap)));

      // No upper clamp beyond that one. w > 1 is Russian roulette on the
      // sites -- fewer and heavier -- and banking stays unbiased at any
      // weight, since E[N] = nu exactly however w is chosen.
      w = std::max(MIN_SITE_WEIGHT, w);
      tree_site_weight[grp.a] = w;
      tree_site_weight[grp.b2] = w;
    }
  }
}

} // namespace

void finalize_batch()
{
  if (!simulation::bep_on)
    return;
  if (!mpi::master) {
    std::fill(batch_tau.begin(), batch_tau.end(), 0.0);
    return;
  }

  int nd = settings::bep_n_generation + 1;

  // One realisation of the estimator per batch, at every depth. The worth is
  // the d = L entry; the rest is the depth-convergence curve, which is what
  // tells the user whether L was deep enough -- the level is flat in d once
  // the perturbed fundamental mode has established itself.
  //
  // The level is formed from the batch's SUMS, not from a mean of
  // per-generation levels: the ratio of means is the estimator, and the mean
  // of ratios is not. generations_per_batch is what makes consecutive batches
  // effectively independent, and it is the user's knob, checked by the
  // plateau of sigma against it.
  size_t np = perturbations.size();
  vector<double> l_b(np * nd, 0.0);
  for (size_t ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];
    for (int d = 0; d < nd; ++d) {
      double l = level(batch_tau[tau_index(p.tree_d, d)],
        batch_tau[tau_index(p.tree_fp, d)], batch_tau[tau_index(p.tree_fn, d)],
        batch_tau[tau_index(p.tree_lp, d)], batch_tau[tau_index(p.tree_ln, d)],
        keff_norm);
      l_b[ip * nd + d] = l;
      ell_sum[ip * nd + d] += l;
    }
  }
  // The full cross-products, not just the squares: two perturbations sharing
  // this run share nearly all of their noise, and the covariance is what lets
  // a difference between them be quoted honestly.
  for (size_t i = 0; i < np; ++i) {
    for (size_t j = 0; j < np; ++j) {
      for (int d = 0; d < nd; ++d) {
        ell_cross[(i * np + j) * nd + d] += l_b[i * nd + d] * l_b[j * nd + d];
      }
    }
  }
  ++n_active_batches;

  // Retune from the batch just closed, before its totals are cleared.
  choose_site_weights();

  // Every per-batch accumulator, cleared together.
  //
  // batch_sources and batch_root_sq used to be missed here, so they were
  // run-cumulative while batch_tau -- which the rule divides them by -- was
  // per-batch. The measured per-root spread then grew without bound as the
  // run went on: rv ~ n*rv_true + (n - 1/n)/m after n active batches, so by
  // a few dozen batches the rule believed every population was hopelessly
  // noisy no matter how finely it was split. Nothing downstream could work:
  // the spread that sets both the floor and the root count was not a spread
  // but a batch counter.
  std::fill(batch_tau.begin(), batch_tau.end(), 0.0);
  std::fill(batch_sources.begin(), batch_sources.end(), 0);
  std::fill(batch_root_sq.begin(), batch_root_sq.end(), 0.0);
  std::fill(batch_pair_sq.begin(), batch_pair_sq.end(), 0.0);
  std::fill(batch_pair_n.begin(), batch_pair_n.end(), 0);
  batch_histories = 0;
}


//==============================================================================
// Output
//==============================================================================

void write_results(hid_t file_id)
{
  if (perturbations.empty() || n_active_batches == 0)
    return;

  int nd = settings::bep_n_generation + 1;
  auto np = static_cast<int>(perturbations.size());
  hid_t group = create_group(file_id, "local_perturbation");

  write_dataset(group, "n_generation", settings::bep_n_generation);
  write_dataset(group, "n_generations_recorded", n_generations);
  write_dataset(group, "n_batches", n_active_batches);
  write_dataset(group, "n_trees", static_cast<int>(tree_pert.size()));
  write_dataset(group, "n_perturbations", np);
  write_dataset(group, "keff", keff_norm);

  // What the rule actually chose, so a figure-of-merit sweep can be plotted
  // against it and so a run that came out noisy can be diagnosed without a
  // rebuild.
  write_dataset(group, "site_weight", tree_site_weight);

  // Root count per tree. Nothing in the Python layer reads it, and it was
  // nearly deleted with n_tracks and n_roots for that reason -- but it is
  // what makes the COST MODEL checkable after the fact. Reconstructing a
  // run's per-generation history budget from n_sources, site_weight and
  // tau_pooled and comparing it against n_histories is what established
  // kappa = L-1 rather than L (docs/bep_autotune.md 11.2), on statepoints
  // that already existed. Instrumentation that can settle a modelling
  // question from data already on disk earns its bytes.
  write_dataset(group, "n_sources", tree_sources);

  // The histories the shadow pass actually transported.
  //
  // WORK, not wall clock. A figure of merit needs a cost, and the cost has to
  // be reproducible or the comparison is not either: two runs of one seed on
  // a loaded box give different seconds and identical work. This is the same
  // quantity the tuning rule optimises against -- histories -- so a measured
  // figure of merit and the rule's own objective are in the same units.
  //
  // A shadow tree transports one history per root plus one per banked site,
  // since every banked site is revived exactly once.
  write_dataset(group, "n_histories", n_history_total);

  vector<int32_t> ids;
  for (const auto& p : perturbations)
    ids.push_back(p.id);
  write_dataset(group, "ids", ids);

  // Everything the analysis needs, and nothing else.
  //
  // `level_sum` and `level_sumsq` are the batch statistics of the estimator
  // itself: per perturbation and per depth, the sum and the sum of squares of
  // the per-batch level over `n_batches` active batches. Mean, sigma and a
  // t-interval on (n_batches - 1) degrees of freedom come out of them the
  // same way they do for any other OpenMC tally -- there is no model of the
  // inter-generation correlation anywhere, because generations_per_batch is
  // what is supposed to remove it.
  //
  // `level_pooled` is the same level formed from the run's POOLED totals
  // instead. It carries no ratio-of-means bias, so the two agreeing to a
  // small fraction of sigma is the one assumption this estimator makes, and
  // it can be checked from the file. If they ever disagree, the pooled one is
  // the number to quote.
  write_dataset(group, "level_sum", ell_sum);
  write_dataset(group, "level_cross", ell_cross);

  vector<double> pooled(static_cast<size_t>(np) * nd, 0.0);
  for (int ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];
    for (int d = 0; d < nd; ++d) {
      pooled[static_cast<size_t>(ip) * nd + d] =
        level(pooled_tau[tau_index(p.tree_d, d)],
          pooled_tau[tau_index(p.tree_fp, d)],
          pooled_tau[tau_index(p.tree_fn, d)],
          pooled_tau[tau_index(p.tree_lp, d)],
          pooled_tau[tau_index(p.tree_ln, d)], keff_norm);
    }
  }
  write_dataset(group, "level_pooled", pooled);

  // The five populations' run totals, [tree][depth]. Kept because they are
  // what a reviewer needs to re-derive the level by hand, and because the
  // relative size of the +/- pair is the only direct read on how much
  // cancellation the signed source is costing. O(n_pert * L) either way.
  write_dataset(group, "tau_pooled", pooled_tau);

  for (int ip = 0; ip < np; ++ip) {
    const Perturbation& p = perturbations[ip];
    hid_t pg = create_group(group, fmt::format("perturbation {}", p.id));
    write_dataset(pg, "index", ip);

    vector<int32_t> trees {p.tree_d, p.tree_fp, p.tree_fn, p.tree_lp,
      p.tree_ln};
    write_dataset(pg, "trees", trees);

    vector<int32_t> cids, mids;
    for (const auto& sub : p.subs) {
      cids.push_back(sub.cell_id);
      mids.push_back(sub.mat_id);
    }
    write_dataset(pg, "cells", cids);
    write_dataset(pg, "materials", mids);

    close_group(pg);
  }

  close_group(group);
}

} // namespace bep

//==============================================================================
// Input
//==============================================================================

void read_perturbations_xml()
{
  // Optional, like tallies.xml.
  std::string filename = settings::path_input + "perturbations.xml";
  if (!file_exists(filename))
    return;

  write_message("Reading perturbations XML file...", 5);

  pugi::xml_document doc;
  doc.load_file(filename.c_str());
  read_perturbations_xml(doc.document_element());
}

void read_perturbations_xml(pugi::xml_node root)
{
  int32_t next_id = 1;
  for (pugi::xml_node node : root.children("local_perturbation")) {
    bep::Perturbation p;
    p.id = check_for_node(node, "id") ? std::stoi(get_node_value(node, "id"))
                                      : next_id;
    next_id = p.id + 1;

    // A perturbation is a SET of substitutions applied together, which is what
    // lets a displacement be expressed as the trailing sliver reverting and
    // the leading sliver taking the sample. The bare <cell>/<material> pair is
    // kept as shorthand for the one-cell case.
    for (pugi::xml_node node_s : node.children("substitution")) {
      bep::Substitution s;
      s.cell_id = std::stoi(get_node_value(node_s, "cell"));
      s.mat_id = std::stoi(get_node_value(node_s, "material"));
      p.subs.push_back(s);
    }
    if (check_for_node(node, "cell")) {
      bep::Substitution s;
      s.cell_id = std::stoi(get_node_value(node, "cell"));
      s.mat_id = std::stoi(get_node_value(node, "material"));
      p.subs.push_back(s);
    }
    bep::perturbations.push_back(p);
  }
}
} // namespace openmc
