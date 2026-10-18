"""Results of the adjoint side populations of an eigenvalue calculation.

Each active generation, three populations of roots are grown a fixed number
of generations beside the driver, in the unperturbed physics, and their
summed weight at every depth is recorded per batch:

- fission roots: a sample of the fission bank, tagged with each site's delayed
  group (0 = prompt);
- delayed roots: the expected delayed neutrons of each group at every driver
  fission-site creation;
- photoneutron roots (optional): photoneutrons from driver photons, taken out
  of the transport so that they never enter the fission chain, tagged with
  the photofission delayed group (0 = prompt).

With perturbed importance on, the fission- and delayed-root trees also
transport photons below their root, and the photoneutrons those photons make
grow on as branches, scored separately. The trees without their branches give
the unperturbed importance; with them, the importance with photoneutrons in
the chain.

A root's weight at depth L is its importance (iterated fission probability).
All roots are weighted like fission sites, per 1/k. With importance sums I
and lifetime-weighted sums T at depth L (primes: branches included), the
quantities reported are

- beta_eff per group g: I_delayed,g / I_fission (forced), or
  I_fission,g / I_fission (analog, as in iterated fission probability);
- generation time: T_fission / (k I_fission);
- photoneutron reactivity: drho = 1/k - 1/k' = I_pn / I_fission, where k' is
  the eigenvalue with photoneutrons transported as secondaries;
- beta_eff with photoneutrons, per group:
  (I'_delayed,g + k I_pn,g) / (I'_fission + k I_pn);
- generation time with photoneutrons:
  (T'_fission + k T_pn) / (k' (I'_fission + k I_pn)),

where a primed sum at depth d includes the branches and is divided by
1 + (d - 1) k drho, the growth the branches add over the d - 1 generations
that make photons (iterated fission probability with photoneutrons in the
chain removes it by dividing by k' at every fission).

The last two follow the convention of iterated fission probability with
photoneutrons in the fission chain: a photoneutron counts as a neutron of the
generation of the neutron that made it, not divided by k. They are first
order in the photoneutron source, including its change of the importance
function (which needs perturbed importance), but not the redistribution of the
fission source that the photoneutrons cause.

Every quantity is a ratio of batch means. Its uncertainty is from the batch
statistics, linearised about the means (the delta method); k is treated as
exact.
"""

import numpy as np
from uncertainties import ufloat

CLASS_FISSION = 0
CLASS_DELAYED = 1
CLASS_PHOTONEUTRON = 2
CLASS_FISSION_BRANCH = 3
CLASS_DELAYED_BRANCH = 4


class AdjointPopulations:
    """Adjoint side-population results read from a statepoint.

    Parameters
    ----------
    group : h5py.Group
        The ``adjoint_populations`` group of a statepoint file
    keff : float
        Combined estimate of k of the calculation
    n_particles : int, optional
        Particles per generation, by which the detector signal is normalized
        like the tallies

    Attributes
    ----------
    n_generation : int
        Depth to which each root was grown
    n_batches : int
        Number of active batches recorded
    photoneutrons : bool
        Whether photoneutrons were taken as a population
    perturbed_importance : bool
        Whether photoneutron branches were grown in the trees
    branchless : bool
        Whether the trees were grown with branchless collisions
    branchless_split : float
        Weight, over the root's, above which a branchless line was split
    keff : float
        k used to form the ratios
    weight : numpy.ndarray
        Summed root-importance weight, indexed [batch, class, tag, depth]
    weight_t0 : numpy.ndarray
        Summed weight times the root's lifetime at the fission that started
        its line, indexed [batch, class, tag, depth]
    n_delayed_groups : int
        Number of delayed groups found in the data
    n_histories : int
        Number of shadow-tree histories tracked in the whole run
    photoneutron_energy_bins : numpy.ndarray or None
        Energy-group edges [eV] of the grouped photoneutron tally, if on
    photoneutron_energy_variable : str or None
        'photon_birth' or 'photoneutron': what the groups bin
    photoneutron_energy_weight : numpy.ndarray or None
        Summed photoneutron-root weight, indexed [batch, energy group, depth]
    photoneutron_fission_nuclides : list of str or None
        Names of the fissioning-nuclide bins, the last being 'other', if the
        photoneutron roots were also tallied by the nuclide whose fission
        made the photon
    photoneutron_nuclide_weight : numpy.ndarray or None
        Summed photoneutron-root weight, indexed [batch, nuclide bin, energy
        group, depth]; one energy group if no energy edges were given
    probe_energies : numpy.ndarray or None
        Probe photon line energies [eV], if probes were on
    probe_fission_nuclides : list of str or None
        Names of the probes' fissioning-nuclide bins ('other' last, or a
        single 'all')
    probe_targets : list of str or None
        Names of the photoneutron-target bins: the nuclides of the setting
        ``photoneutron_targets`` and 'other', or a single 'all'
    probe_weight : numpy.ndarray or None
        Summed probe-root weight, indexed [batch, nuclide bin, line, label,
        depth], summed over the photoneutron targets. With three labels: 0
        uncollided (empty when rays were on), 1 after coherent scattering
        only, 2 after an energy-changing collision. Files written before the
        labels have two: 0 at the line energy (uncollided or coherent), 1
        after an energy-changing collision
    probe_weight_by_target : numpy.ndarray or None
        As probe_weight, by photoneutron target: [batch, nuclide bin, line,
        target, label, depth]
    probe_n_labels : int or None
        Number of probe labels (2 or 3)
    ray_weight : numpy.ndarray or None
        Summed uncollided photoneutron-root weight of the ray probes,
        indexed [batch, nuclide bin, ray line, depth], summed over the
        photoneutron targets, if rays were on
    ray_weight_by_target : numpy.ndarray or None
        As ray_weight, by photoneutron target: [batch, nuclide bin, ray line,
        target, depth]
    ray_photon_energies : numpy.ndarray or None
        The rays' line energies [eV]: the probe lines, each interval
        subdivided ray_refinement times
    ray_refinement : int
        Subdivisions of each probe-line interval on the ray lines (1: the ray
        lines are the probe lines)
    ray_comb_weight : numpy.ndarray or None
        The ray trees' summed weight by photoneutron comb energy, indexed
        [batch, nuclide bin, comb energy, depth]
    ray_neutron_energies : numpy.ndarray or None
        The photoneutron comb [eV]
    ray_fission_weight : numpy.ndarray or None
        Fission weight of the events the rays were cast from, [batch,
        nuclide bin]
    probe_fission_weight : numpy.ndarray or None
        Probed fission weight (sum of w/k sigma_f/sigma_t over the recorded
        fission events), indexed [batch, nuclide bin]: the weight of probe
        photons emitted per line
    detector_source : numpy.ndarray or None
        Source weight of the detector trees' categories, indexed [batch,
        category, origin bin], if detector tallies were on: the fission-bank
        weight of prompt sites, the delayed-neutron weight per group, the
        driver photoneutrons' weight, and the fission weight emitting each
        probe and ray line. The detector tallies are divided by it per
        generation.
    detector_categories : tuple of int or None
        The first category of the driver photoneutrons, the probes and the
        rays, and the number of categories. Categories 0 and 1 to 8 are
        prompt fission neutrons and the delayed groups.
    n_detector_histories : int
        Histories transported in detector trees
    detector_batch_values : dict of int to numpy.ndarray
        Each detector tally's values per batch, by tally ID, indexed [batch,
        filter bin x nuclide x score] (flattened as in the tally)
    probe_trigger : tuple of float or None
        (threshold, floor) of the probe-importance trigger, if one was set

    """

    def __init__(self, group, keff, n_particles=None):
        self.n_generation = int(group['n_generation'][()])
        self.n_batches = int(group['n_batches'][()])
        self.photoneutrons = bool(group['photoneutrons'][()])
        self.perturbed_importance = bool(group['perturbed_importance'][()]) \
            if 'perturbed_importance' in group else False
        self.branchless = bool(group['branchless'][()]) \
            if 'branchless' in group else False
        self.branchless_split = float(group['branchless_split'][()]) \
            if 'branchless_split' in group else 0.0
        n_class = int(group['n_class'][()])
        n_tag = int(group['n_tag'][()])
        shape = (self.n_batches, n_class, n_tag, self.n_generation + 1)
        self.weight = group['weight'][()].reshape(shape)
        self.weight_t0 = group['weight_t0'][()].reshape(shape)
        self.site_weight = group['site_weight'][()]
        self.n_roots_last_generation = group['n_roots_last_generation'][()]
        self.raw_weight_last_generation = \
            group['raw_weight_last_generation'][()]
        self.n_histories = int(group['n_histories'][()])
        self.keff = float(keff)

        self.photoneutron_energy_bins = None
        self.photoneutron_energy_variable = None
        self.photoneutron_energy_weight = None
        if 'photoneutron_energy_bins' in group:
            edges = group['photoneutron_energy_bins'][()]
            self.photoneutron_energy_bins = edges
            var = group['photoneutron_energy_variable'][()]
            self.photoneutron_energy_variable = \
                var.decode() if isinstance(var, bytes) else str(var)
            self.photoneutron_energy_weight = \
                group['photoneutron_energy_weight'][()].reshape(
                    (self.n_batches, len(edges) - 1, self.n_generation + 1))

        self.photoneutron_fission_nuclides = None
        self.photoneutron_nuclide_weight = None
        if 'photoneutron_fission_nuclides' in group:
            names = group['photoneutron_fission_nuclides'][()]
            names = names.decode() if isinstance(names, bytes) else str(names)
            self.photoneutron_fission_nuclides = names.split()
            n_group = 1 if self.photoneutron_energy_bins is None else \
                len(self.photoneutron_energy_bins) - 1
            self.photoneutron_nuclide_weight = \
                group['photoneutron_nuclide_weight'][()].reshape(
                    (self.n_batches, len(self.photoneutron_fission_nuclides),
                     n_group, self.n_generation + 1))

        self.probe_energies = None
        self.probe_n_labels = None
        self.probe_fission_nuclides = None
        self.probe_targets = None
        self.probe_weight = None
        self.probe_weight_by_target = None
        self.probe_fission_weight = None
        if 'probe_energies' in group:
            self.probe_energies = group['probe_energies'][()]
            names = group['probe_fission_nuclides'][()]
            names = names.decode() if isinstance(names, bytes) else str(names)
            self.probe_fission_nuclides = names.split()
            nn = len(self.probe_fission_nuclides)
            self.probe_n_labels = int(group['probe_n_labels'][()]) \
                if 'probe_n_labels' in group else 2
            if 'probe_targets' in group:
                t = group['probe_targets'][()]
                self.probe_targets = (t.decode() if isinstance(t, bytes)
                                      else str(t)).split()
            else:
                self.probe_targets = ['all']
            self.probe_weight_by_target = group['probe_weight'][()].reshape(
                (self.n_batches, nn, len(self.probe_energies),
                 len(self.probe_targets), self.probe_n_labels,
                 self.n_generation + 1))
            self.probe_weight = self.probe_weight_by_target.sum(axis=3)
            self.probe_fission_weight = \
                group['probe_fission_weight'][()].reshape(self.n_batches, nn)
        self.probe_trigger = tuple(float(x) for x in group['probe_trigger'][()]) \
            if 'probe_trigger' in group else None
        self.ray_weight = None
        self.ray_weight_by_target = None
        self.ray_comb_weight = None
        self.ray_neutron_energies = None
        self.ray_fission_weight = None
        self.ray_photon_energies = None
        self.ray_refinement = 1
        if 'ray_weight' in group:
            nn = len(self.probe_fission_nuclides)
            nd = self.n_generation + 1
            self.ray_neutron_energies = group['ray_neutron_energies'][()]
            self.ray_photon_energies = group['ray_photon_energies'][()] \
                if 'ray_photon_energies' in group else self.probe_energies
            self.ray_refinement = int(group['ray_refinement'][()]) \
                if 'ray_refinement' in group else 1
            self.ray_weight_by_target = group['ray_weight'][()].reshape(
                (self.n_batches, nn, len(self.ray_photon_energies),
                 len(self.probe_targets), nd))
            self.ray_weight = self.ray_weight_by_target.sum(axis=3)
            self.ray_comb_weight = group['ray_comb_weight'][()].reshape(
                (self.n_batches, nn, len(self.ray_neutron_energies), nd))
            self.ray_fission_weight = \
                group['ray_fission_weight'][()].reshape(self.n_batches, nn)

        self.detector_source = None
        self.detector_categories = None
        self.n_detector_histories = 0
        if 'detector_source' in group:
            cats = tuple(int(c) for c in group['detector_categories'][()])
            n_origin = int(group['detector_n_origin_bins'][()])
            self.detector_categories = cats
            self.detector_source = group['detector_source'][()].reshape(
                (self.n_batches, cats[3], n_origin))
            self.n_detector_histories = int(group['n_detector_histories'][()])
        self.n_particles = n_particles
        self.detector_batch_values = {}
        if 'detector_tally_ids' in group:
            for tid in group['detector_tally_ids'][()]:
                values = group[f'detector_tally_{int(tid)}'][()]
                self.detector_batch_values[int(tid)] = values.reshape(
                    self.n_batches, -1)

        # The highest group any fission or delayed root was tagged with
        w = self.weight[:, :CLASS_PHOTONEUTRON].sum(axis=(0, 1, 3))
        nonzero = np.nonzero(w[1:])[0]
        self.n_delayed_groups = int(nonzero[-1] + 1) if nonzero.size else 0

    # -------------------------------------------------------------------------
    # Per-batch sums at one depth

    def _depth(self, depth):
        d = self.n_generation if depth is None else depth
        if not 1 <= d <= self.n_generation:
            raise ValueError(f'depth must be in [1, {self.n_generation}].')
        return d

    def _groups(self, group):
        if group is None:
            return slice(1, None)
        if not 1 <= group <= self.weight.shape[2] - 1:
            raise ValueError(f'Invalid delayed group {group}.')
        return slice(group, group + 1)

    def _sums(self, depth, group):
        d = self._depth(depth)
        g = self._groups(group)
        w = self.weight[..., d]
        wt = self.weight_t0[..., d]
        x = {
            'I_F': w[:, CLASS_FISSION].sum(axis=1),
            'I_Fg': w[:, CLASS_FISSION, g].sum(axis=1),
            'I_Dg': w[:, CLASS_DELAYED, g].sum(axis=1),
            'I_P': w[:, CLASS_PHOTONEUTRON].sum(axis=1),
            'I_Pg': w[:, CLASS_PHOTONEUTRON, g].sum(axis=1),
            'T_F': wt[:, CLASS_FISSION].sum(axis=1),
            'T_P': wt[:, CLASS_PHOTONEUTRON].sum(axis=1),
        }
        if w.shape[1] > CLASS_DELAYED_BRANCH:
            x['B_F'] = w[:, CLASS_FISSION_BRANCH].sum(axis=1)
            x['B_Dg'] = w[:, CLASS_DELAYED_BRANCH, g].sum(axis=1)
            x['BT_F'] = wt[:, CLASS_FISSION_BRANCH].sum(axis=1)
        return x

    def _estimate(self, func, depth, group=None):
        """Evaluate func at the batch means of the sums, with a delta-method
        standard deviation from the batch-to-batch spread."""
        x = self._sums(depth, group)
        keys = list(x)
        n = self.n_batches
        m = {k: v.mean() for k, v in x.items()}
        value = func(m)
        if n < 2:
            return ufloat(value, np.nan)

        # Gradient by central differences, scaled to each mean
        z = np.zeros(n)
        for k in keys:
            if np.all(x[k] == m[k]):
                continue
            h = 1e-6 * abs(m[k])
            up = dict(m)
            dn = dict(m)
            up[k] += h
            dn[k] -= h
            grad = (func(up) - func(dn)) / (2.0 * h)
            if grad != 0.0:
                z += grad * (x[k] - m[k])
        return ufloat(value, z.std(ddof=1) / np.sqrt(n))

    # -------------------------------------------------------------------------
    # Unperturbed kinetics parameters

    def beta_eff(self, group=None, forced=True, depth=None):
        """Effective delayed-neutron fraction of the unperturbed system.

        Parameters
        ----------
        group : int, optional
            Delayed group (1-based); all groups if None
        forced : bool
            Use the expected-value delayed roots (True) or the delayed sites
            that happen to be in the fission bank (False, as in iterated
            fission probability)
        depth : int, optional
            Depth at which the importance is read; n_generation if None

        Returns
        -------
        uncertainties.UFloat

        """
        num = 'I_Dg' if forced else 'I_Fg'
        return self._estimate(lambda m: m[num] / m['I_F'], depth, group)

    def generation_time(self, depth=None):
        """Generation time of the unperturbed system [s]."""
        k = self.keff
        return self._estimate(lambda m: m['T_F'] / (k * m['I_F']), depth)

    # -------------------------------------------------------------------------
    # Photoneutron effects

    def _check_photoneutrons(self):
        if not self.photoneutrons:
            raise ValueError('The calculation did not take photoneutrons.')

    def _check_energy_groups(self):
        if self.photoneutron_energy_weight is None:
            raise ValueError('The calculation did not tally photoneutrons '
                             'by energy group.')

    def photoneutron_reactivity_by_energy(self, depth=None):
        """Contribution of each energy group to the photoneutron reactivity,
        I_pn,e / I_fission; they sum to :meth:`photoneutron_reactivity` when
        the groups cover every photoneutron.

        Returns
        -------
        list of uncertainties.UFloat
        """
        self._check_photoneutrons()
        self._check_energy_groups()
        d = self._depth(depth)
        i_f = self.weight[:, CLASS_FISSION, :, d].sum(axis=1)
        out = []
        for e in range(self.photoneutron_energy_weight.shape[1]):
            i_e = self.photoneutron_energy_weight[:, e, d]
            out.append(_ratio(i_e, i_f))
        return out

    def photoneutron_importance_by_energy(self, depth=None):
        """Importance per unit weight of a photoneutron in each energy group
        relative to a fission neutron's: (I_pn,e(d)/I_pn,e(0)) /
        (I_fission(d)/I_fission(0)). NaN for an empty group.

        Returns
        -------
        list of uncertainties.UFloat
        """
        self._check_photoneutrons()
        self._check_energy_groups()
        d = self._depth(depth)
        w = self.weight[:, CLASS_FISSION].sum(axis=1)
        out = []
        for e in range(self.photoneutron_energy_weight.shape[1]):
            x = self.photoneutron_energy_weight[:, e]
            if not x[:, 0].any():
                out.append(ufloat(np.nan, np.nan))
                continue
            out.append(_double_ratio(x[:, d], x[:, 0], w[:, d], w[:, 0]))
        return out

    def _check_nuclides(self):
        if self.photoneutron_nuclide_weight is None:
            raise ValueError('The calculation did not tally photoneutrons by '
                             'fissioning nuclide.')

    def photoneutron_reactivity_by_nuclide(self, depth=None):
        """Contribution of the photons from each fissioning nuclide (and
        energy group) to the photoneutron reactivity, I_pn,j,e / I_fission.

        Returns
        -------
        dict of str to list of uncertainties.UFloat
            One list per nuclide bin (the last is 'other'), one entry per
            energy group (a single entry if no energy edges were given)
        """
        self._check_photoneutrons()
        self._check_nuclides()
        d = self._depth(depth)
        i_f = self.weight[:, CLASS_FISSION, :, d].sum(axis=1)
        w = self.photoneutron_nuclide_weight
        return {name: [_ratio(w[:, j, e, d], i_f) for e in range(w.shape[2])]
                for j, name in enumerate(self.photoneutron_fission_nuclides)}

    def photoneutron_importance_by_nuclide(self, depth=None):
        """Importance per unit weight of a photoneutron from each fissioning
        nuclide's photons (and energy group), relative to a fission
        neutron's. NaN for an empty bin.

        Returns
        -------
        dict of str to list of uncertainties.UFloat
        """
        self._check_photoneutrons()
        self._check_nuclides()
        d = self._depth(depth)
        wf = self.weight[:, CLASS_FISSION].sum(axis=1)
        w = self.photoneutron_nuclide_weight
        out = {}
        for j, name in enumerate(self.photoneutron_fission_nuclides):
            row = []
            for e in range(w.shape[2]):
                x = w[:, j, e]
                if not x[:, 0].any():
                    row.append(ufloat(np.nan, np.nan))
                    continue
                row.append(_double_ratio(x[:, d], x[:, 0], wf[:, d], wf[:, 0]))
            out[name] = row
        return out

    def _check_probes(self):
        if self.probe_weight is None:
            raise ValueError('The calculation did not emit probe photons: '
                             "set adjoint_populations"
                             "['photoneutron_probe_energies'].")

    def _probe_bin(self, nuclide):
        names = self.probe_fission_nuclides
        if nuclide is None:
            if len(names) != 1:
                raise ValueError(f'Name a probe nuclide bin: {names}.')
            return 0
        if nuclide not in names:
            raise ValueError(f'No probe nuclide bin {nuclide!r}: {names}.')
        return names.index(nuclide)

    def _probe_parts(self, j, d):
        """Direct and scattered probe-root weight per batch and line at
        depth d: with rays, direct = rays + coherent-only probes."""
        w = self.probe_weight[:, j, :, :, d]
        if self.probe_n_labels == 2:
            return w[:, :, 0], w[:, :, 1]
        direct = w[:, :, 0] + w[:, :, 1]
        if self.ray_weight is not None:
            # The probe lines are every ray_refinement-th ray line
            direct = self.ray_weight[:, j, ::self.ray_refinement, d] + \
                w[:, :, 1]
        return direct, w[:, :, 2]

    def probe_photoneutron_yield(self, nuclide=None):
        """Photoneutrons made per probe photon of each line (depth 0), split
        into those made at the line energy and after scattering.

        Returns
        -------
        tuple of list of uncertainties.UFloat
            (direct, scattered), one entry per line

        """
        self._check_probes()
        j = self._probe_bin(nuclide)
        f = self.probe_fission_weight[:, j]
        out = []
        for part in self._probe_parts_yield(j):
            out.append([_ratio(part[:, k], f) for k in range(part.shape[1])])
        return tuple(out)

    def _probe_parts_yield(self, j):
        """As _probe_parts at depth 0, per unit fission weight of the probes;
        the rays' part is rescaled from their own fission weight."""
        if self.ray_weight is None or self.probe_n_labels == 2:
            return self._probe_parts(j, 0)
        w = self.probe_weight[:, j, :, :, 0]
        scale = (self.probe_fission_weight[:, j] /
                 np.where(self.ray_fission_weight[:, j] > 0,
                          self.ray_fission_weight[:, j], 1.0))[:, None]
        ray = self.ray_weight[:, j, ::self.ray_refinement, 0]
        return ray * scale + w[:, :, 1], w[:, :, 2]

    def probe_reactivity(self, nuclide=None, depth=None):
        """Photoneutron reactivity of each probe line of unit intensity per
        fission of a nuclide, I_probe / I_fission, split into direct and
        scattered parts.

        Returns
        -------
        tuple of list of uncertainties.UFloat
            (direct, scattered), one entry per line

        """
        self._check_probes()
        j = self._probe_bin(nuclide)
        d = self._depth(depth)
        i_f = self.weight[:, CLASS_FISSION, :, d].sum(axis=1)
        return tuple([_ratio(part[:, k], i_f) for k in range(part.shape[1])]
                     for part in self._probe_parts(j, d))

    def _line_totals(self, j, d):
        """Direct plus scattered weight per batch at the trigger's lines: the
        ray lines, the probes' parts interpolated per batch from the probe
        lines with the subdivision weights; the probe lines without rays."""
        direct, scattered = self._probe_parts(j, d)
        if self.ray_weight is None or self.probe_n_labels == 2:
            return direct + scattered
        w = self.probe_weight[:, j, :, :, d]
        collided = w[:, :, 1] + w[:, :, 2]
        r = self.ray_refinement
        n = self.ray_photon_energies.size
        i = np.arange(n)
        k = i // r
        t = (i % r) / r
        k1 = np.minimum(k + 1, collided.shape[1] - 1)
        return (self.ray_weight[:, j, :, d] + collided[:, k] * (1.0 - t) +
                collided[:, k1] * t)

    def ray_comb_importance(self, nuclide=None, depth=None):
        """Importance per unit weight of a photoneutron started at each
        energy of the rays' neutron comb, relative to a fission neutron's:
        (I_c(d)/I_c(0)) / (I_F(d)/I_F(0)), pooled over the nuclide bins if
        none is named. The comb's linear interpolation is checked against it.

        Returns
        -------
        list of uncertainties.UFloat
            One entry per comb energy (NaN where no tree started)
        """
        if self.ray_comb_weight is None:
            raise ValueError('The calculation did not use ray probes.')
        d = self._depth(depth)
        w = self.weight[:, CLASS_FISSION].sum(axis=1)
        if nuclide is None:
            c = self.ray_comb_weight.sum(axis=1)
        else:
            c = self.ray_comb_weight[:, self._probe_bin(nuclide)]
        out = []
        for m in range(c.shape[1]):
            if not c[:, m, 0].any():
                out.append(ufloat(np.nan, np.nan))
                continue
            out.append(_double_ratio(c[:, m, d], c[:, m, 0], w[:, d], w[:, 0]))
        return out

    def probe_relative_error(self, nuclide=None, depth=None, floor=0.1):
        """Standard deviation of each line's importance (direct plus
        scattered, per unit fission-neutron importance) over
        max(importance, floor * peak): the relative error where the
        importance is at least floor of its peak, the error relative to
        floor of the peak below. The probe trigger holds this below its
        threshold at every line. With rays the lines are the ray lines, the
        probes' collided part interpolated per batch from the probe lines.

        Returns
        -------
        numpy.ndarray
            One value per line (ray line with rays)
        """
        self._check_probes()
        j = self._probe_bin(nuclide)
        d = self._depth(depth)
        i_f = self.weight[:, CLASS_FISSION, :, d].sum(axis=1)
        r = [_ratio(x, i_f) for x in self._line_totals(j, d).T]
        mean = np.array([x.n for x in r])
        std = np.array([x.s for x in r])
        peak = mean.max()
        if not peak > 0.0:
            return np.full(mean.shape, np.inf)
        return std / np.maximum(mean, floor * peak)

    def probe_importance(self, target, nuclide=None, depth=None, rtol=0.01,
                         floor=1.0e-3, e_max=None, cross_sections=None):
        """Importance of a photon born at a fission of a nuclide, rebuilt at
        every energy from the probe lines and refined for linear
        interpolation.

        With ray probes on, the direct part is the rays' uncollided
        photoneutrons (one shared tree for every line) plus the probes'
        coherently scattered ones, and the scattered part the probes'
        photoneutrons after an energy-changing collision.

        The direct part of each line is interpolated divided by the target's
        photoneutron production over its removal cross section, and the
        scattered part times the removal cross section, both linearly in
        photoneutron lethargy; the two are recombined with the exact cross
        sections (:mod:`openmc.photoneutron_probes`). The result is tabulated
        on a grid on which linear interpolation is within rtol of the rebuilt
        function.

        Parameters
        ----------
        target : openmc.PhotoneutronTarget or openmc.Material, or dict
            Cross sections of the medium that makes the photoneutrons, or the
            material itself (its cross sections are then formed with
            :meth:`openmc.PhotoneutronTarget.from_material`). With the
            photoneutrons tallied by target (the setting
            ``photoneutron_targets``, e.g. ['Be9', 'H2']), a dict from each
            target bin to its medium, e.g. {'Be9': beryllium, 'H2':
            heavy_water}: a material then counts only that nuclide's
            photoneutrons (the 'other' bin's, all but the named targets'), and
            every target is rebuilt with its own cross sections and threshold.
            The 'other' bin may be left out; its share is then reported in a
            warning and dropped. A single target with the photoneutrons
            tallied by target takes their sum.
        nuclide : str, optional
            Fissioning-nuclide bin; may be omitted if there is only one
        depth : int, optional
            Depth at which the importance is read; n_generation if None
        rtol : float
            Tolerance of linear interpolation of the table
        floor : float
            Fraction of the largest importance below which the tolerance is
            absolute
        e_max : float, optional
            Top of the table [eV]; the highest line if None
        cross_sections : str or os.PathLike, optional
            cross_sections.xml for a material target;
            openmc.config['cross_sections'] if None

        Returns
        -------
        openmc.ProbeImportance
            Its table (``energy``, ``mean``, ``std_dev``) is linearly
            interpolable within rtol; calling it interpolates the table

        """
        from openmc.photoneutron_probes import ProbeImportance
        self._check_probes()
        target = self._resolve_targets(target, cross_sections)
        j = self._probe_bin(nuclide)
        d = self._depth(depth)
        i_f = self.weight[:, CLASS_FISSION, :, d].sum(axis=1)
        name = self.probe_fission_nuclides[j]
        rays = self.ray_weight is not None and self.probe_n_labels == 3
        if isinstance(target, dict):
            direct, scattered, ray_direct = {}, {}, {}
            for key in target:
                t = self.probe_targets.index(key)
                w = self.probe_weight_by_target[:, j, :, t, :, d]
                if rays:
                    direct[key], scattered[key] = w[:, :, 1], w[:, :, 2]
                    ray_direct[key] = self.ray_weight_by_target[:, j, :, t, d]
                elif self.probe_n_labels == 2:
                    direct[key], scattered[key] = w[:, :, 0], w[:, :, 1]
                else:
                    direct[key] = w[:, :, 0] + w[:, :, 1]
                    scattered[key] = w[:, :, 2]
            return ProbeImportance(
                self.probe_energies, direct, scattered, i_f, target,
                rtol=rtol, floor=floor, e_max=e_max, name=name,
                ray_lines=self.ray_photon_energies if rays else None,
                ray_direct=ray_direct if rays else None)
        if rays:
            # The rays on their own lines, the probes' parts on theirs
            w = self.probe_weight[:, j, :, :, d]
            return ProbeImportance(self.probe_energies, w[:, :, 1], w[:, :, 2],
                                   i_f, target, rtol=rtol, floor=floor,
                                   e_max=e_max, name=name,
                                   ray_lines=self.ray_photon_energies,
                                   ray_direct=self.ray_weight[:, j, :, d])
        direct, scattered = self._probe_parts(j, d)
        return ProbeImportance(self.probe_energies, direct, scattered,
                               i_f, target, rtol=rtol, floor=floor,
                               e_max=e_max, name=name)

    def probe_importances(self, target, depth=None, rtol=0.01, floor=1.0e-3,
                          e_max=None, cross_sections=None):
        """:meth:`probe_importance` for every fissioning-nuclide bin.

        Returns
        -------
        dict of str to openmc.ProbeImportance
            One linearly interpolable table per nuclide bin

        """
        self._check_probes()
        target = self._resolve_targets(target, cross_sections)
        return {n: self.probe_importance(target, nuclide=n, depth=depth,
                                         rtol=rtol, floor=floor, e_max=e_max)
                for n in self.probe_fission_nuclides}

    def _resolve_targets(self, target, cross_sections=None):
        """The photoneutron target(s) of probe_importance: one
        PhotoneutronTarget, or a dict from target bin to PhotoneutronTarget
        when the photoneutrons were tallied by target."""
        import warnings
        from openmc.photoneutron_probes import PhotoneutronTarget

        def single(t):
            return t if isinstance(t, PhotoneutronTarget) else \
                PhotoneutronTarget.from_material(t, cross_sections)

        if not isinstance(target, dict):
            return single(target)
        bins = self.probe_targets
        if bins == ['all']:
            if len(target) != 1:
                raise ValueError(
                    'The calculation did not tally the photoneutrons by '
                    "target: set adjoint_populations['photoneutron_targets'] "
                    'to name them, or give a single target.')
            return single(next(iter(target.values())))
        named = [b for b in bins if b != 'other']
        unknown = set(target) - set(bins)
        if unknown:
            raise ValueError(f'No photoneutron target bin {sorted(unknown)}: '
                             f'{bins}.')
        missing = [b for b in named if b not in target]
        if missing:
            raise ValueError(f'No medium given for the photoneutron target '
                             f'bins {missing}.')
        out = {}
        for b in bins:
            if b not in target:
                # 'other' left out: report its share of the probe weight
                t = bins.index(b)
                w = self.probe_weight_by_target[..., t, :, :].sum()
                if self.ray_weight_by_target is not None:
                    w += self.ray_weight_by_target[..., t, :].sum()
                total = self.probe_weight.sum() + (
                    0.0 if self.ray_weight is None else self.ray_weight.sum())
                if w > 0.0:
                    warnings.warn(
                        f"The 'other' photoneutron target bin holds "
                        f"{w / total:.3%} of the probe weight and no medium was "
                        'given for it: it is left out.')
                continue
            t = target[b]
            if not isinstance(t, PhotoneutronTarget):
                if b == 'other':
                    t = PhotoneutronTarget.from_material(
                        t, cross_sections, exclude=named)
                else:
                    t = PhotoneutronTarget.from_material(
                        t, cross_sections, nuclides=[b])
            out[b] = t
        return out

    def export_photon_importance(self, path, target, depth=None, rtol=1.0e-3,
                                 floor=1.0e-3, e_max=None,
                                 cross_sections=None):
        """Write the photon importance of every fissioning-nuclide bin and
        the bins' fission fractions to an HDF5 file, for folding with
        delayed-photon sources outside OpenMC (e.g. glowup).

        The file holds, at the root, the attributes ``filetype``
        ('photon_importance'), ``version`` (1, 0), ``nuclides`` (the bin
        names, 'other' for every fissioning nuclide not named), ``depth``,
        ``n_batches``, ``keff`` and ``threshold`` (of the photoneutron target,
        the lowest of the targets' [eV]); the datasets ``fission_importance`` [batch], the fission-root
        importance I_F of each batch, ``delayed_importance`` [batch, group],
        the delayed-root importance of each delayed group, and
        ``fission_lifetime`` [batch], the lifetime-weighted fission-root sum
        T_F, so that beta_eff = mean(delayed_importance) / mean(I_F) and the
        generation time mean(T_F) / (k mean(I_F)) pair with the importance
        batch by batch; and one group per bin, ``nuclides/<name>``, with

        - ``energy`` [eV]: a grid on which linear interpolation of the
          importance is within rtol of the rebuilt function;
        - ``importance``, ``std_dev``: the importance I_j(E) on the grid,
          the photoneutron reactivity (per unit fission-neutron importance)
          that a photon line of unit intensity per fission of the bin's
          nuclides adds, and its batch standard deviation;
        - ``batch_numerator`` [batch, energy]: its per-batch numerators, so
          that I_j(E) = mean(batch_numerator) / mean(fission_importance)
          over any set of batches;
        - ``fission_weight`` [batch]: the collision estimate of the bin's
          fissions per batch, with the attribute ``fission_fraction``, the
          bin's share of all fissions.

        A delayed-photon source S_j(E) per fission of bin j then gives
        beta_ph = sum_j integral S_j(E) I_j(E) dE; I_j / fission_fraction is
        the importance of one photon born at an average fission of the bin.

        Parameters
        ----------
        path : str or os.PathLike
            File to write
        target : openmc.PhotoneutronTarget or openmc.Material, or dict
            The medium (or, by photoneutron target, the media) that makes
            the photoneutrons, as for :meth:`probe_importance`
        depth : int, optional
            Depth at which the importance is read; n_generation if None
        rtol : float
            Tolerance of linear interpolation of the tables
        floor : float
            Fraction of the largest importance below which the tolerance is
            absolute
        e_max : float, optional
            Top of the tables [eV]; the highest line if None
        cross_sections : str or os.PathLike, optional
            cross_sections.xml for a material target

        """
        import h5py
        from openmc import __version__
        self._check_probes()
        target = self._resolve_targets(target, cross_sections)
        d = self._depth(depth)
        tables = self.probe_importances(target, depth=d, rtol=rtol,
                                        floor=floor, e_max=e_max)
        names = self.probe_fission_nuclides
        fw = self.probe_fission_weight
        total = fw.sum()
        with h5py.File(path, 'w') as f:
            f.attrs['filetype'] = np.bytes_('photon_importance')
            f.attrs['version'] = np.array([1, 0])
            f.attrs['openmc_version'] = np.bytes_(__version__)
            f.attrs['nuclides'] = np.array([n.encode() for n in names])
            f.attrs['depth'] = d
            f.attrs['n_batches'] = self.n_batches
            f.attrs['keff'] = self.keff
            f.attrs['threshold'] = tables[names[0]].threshold
            f.attrs['rtol'] = rtol
            f.attrs['floor'] = floor
            f['fission_importance'] = tables[names[0]]._fission
            f['delayed_importance'] = self.weight[:, CLASS_DELAYED, 1:, d]
            f['fission_lifetime'] = \
                self.weight_t0[:, CLASS_FISSION, :, d].sum(axis=1)
            for j, name in enumerate(names):
                imp = tables[name]
                g = f.create_group(f'nuclides/{name}')
                g['energy'] = imp.energy
                g['importance'] = imp.mean
                g['std_dev'] = imp.std_dev
                g['batch_numerator'] = imp._batch_numerators(imp.energy)
                g['fission_weight'] = fw[:, j]
                g.attrs['fission_fraction'] = \
                    fw[:, j].sum() / total if total > 0 else 0.0

    def photoneutron_reactivity(self, depth=None):
        """Reactivity added by the photoneutrons, 1/k - 1/k'."""
        self._check_photoneutrons()
        return self._estimate(lambda m: m['I_P'] / m['I_F'], depth)

    def keff_with_photoneutrons(self, depth=None):
        """k' of the system with photoneutrons transported."""
        self._check_photoneutrons()
        k = self.keff
        return self._estimate(
            lambda m: k / (1.0 - k * m['I_P'] / m['I_F']), depth)

    def _check_perturbed(self):
        self._check_photoneutrons()
        if not self.perturbed_importance:
            raise ValueError(
                'The change of beta_eff and of the generation time needs the '
                'change of the importance function: run with '
                "adjoint_populations['perturbed_importance'] = True.")

    def _perturbed(self, m, d):
        """Importance sums with photoneutron branches, less the growth the
        branches add to every tree. Branches do not branch, so over the d - 1
        generations that make photons a tree grows by 1 + (d - 1) k drho to
        first order; iterated fission probability with photoneutrons in the
        chain removes that growth by dividing by k' at every fission."""
        k = self.keff
        g = 1.0 + (d - 1) * k * m['I_P'] / m['I_F']
        return ((m['I_F'] + m['B_F']) / g, (m['I_Dg'] + m['B_Dg']) / g,
                (m['T_F'] + m['BT_F']) / g)

    def _beta_pn(self, m, d):
        k = self.keff
        I_F, I_D, _ = self._perturbed(m, d)
        return (I_D + k * m['I_Pg']) / (I_F + k * m['I_P'])

    def _gen_time_pn(self, m, d):
        k = self.keff
        kp = k / (1.0 - k * m['I_P'] / m['I_F'])
        I_F, _, T_F = self._perturbed(m, d)
        return (T_F + k * m['T_P']) / (kp * (I_F + k * m['I_P']))

    def beta_eff_with_photoneutrons(self, group=None, depth=None):
        """Effective delayed-neutron fraction with photoneutrons transported,
        using the expected-value delayed roots."""
        self._check_perturbed()
        d = self._depth(depth)
        return self._estimate(lambda m: self._beta_pn(m, d), depth, group)

    def delta_beta_eff(self, group=None, depth=None):
        """Change in beta_eff due to the photoneutrons."""
        self._check_perturbed()
        d = self._depth(depth)
        return self._estimate(
            lambda m: self._beta_pn(m, d) - m['I_Dg'] / m['I_F'], depth, group)

    def generation_time_with_photoneutrons(self, depth=None):
        """Generation time with photoneutrons transported [s]."""
        self._check_perturbed()
        d = self._depth(depth)
        return self._estimate(lambda m: self._gen_time_pn(m, d), depth)

    def delta_generation_time(self, depth=None):
        """Change in generation time due to the photoneutrons [s]."""
        self._check_perturbed()
        k = self.keff
        d = self._depth(depth)
        return self._estimate(
            lambda m: self._gen_time_pn(m, d) - m['T_F'] / (k * m['I_F']),
            depth)

    # -------------------------------------------------------------------------

    def detector_response(self, tally, score=0, nuclide=0, bins=None):
        """Per-batch detector responses of a detector tally, by root category
        and origin, summed over the selected bins of its other filters.

        Parameters
        ----------
        tally : openmc.Tally
            A detector tally read from the same statepoint
        score : int or str
            Score, by index or name
        nuclide : int or str
            Nuclide, by index or name
        bins : dict, optional
            Bins to keep of the tally's other filters, summed; filters not
            listed are summed over all their bins. Keys are a filter of the
            tally, its class or its ID; values are a bin index, a sequence of
            indices, or a boolean mask over the filter's bins. For a
            :class:`openmc.MeshFilter` the mask may also have the mesh's
            shape (``mesh.dimension``), e.g. ``mesh.centroids`` tested for a
            region. For example, the thermal group of an energy filter in a
            ring of a cylindrical mesh::

                r = mesh_filter.mesh.centroids[..., 0]   # for a RegularMesh
                pops.detector_response(tally, bins={
                    openmc.EnergyFilter: 0,
                    openmc.MeshFilter: (r > 60.0) & (r < 77.0)})

        Returns
        -------
        openmc.DetectorResponse

        """
        import openmc
        if tally.id not in self.detector_batch_values:
            raise ValueError(f'Tally {tally.id} has no per-batch detector '
                             'values in the statepoint.')
        if isinstance(score, str):
            score = tally.scores.index(score)
        if isinstance(nuclide, str):
            nuclide = tally.get_nuclide_index(nuclide)
        values = self.detector_batch_values[tally.id].reshape(
            (self.n_batches, tally.num_filter_bins, tally.num_nuclides,
             tally.num_scores))[:, :, nuclide, score]
        shape = [f.num_bins for f in tally.filters]
        values = values.reshape([self.n_batches] + shape)
        i_root = [isinstance(f, openmc.AdjointRootFilter)
                  for f in tally.filters].index(True)
        for key, sel in (bins or {}).items():
            j = self._filter_index(tally, key)
            if j == i_root:
                raise ValueError('The AdjointRootFilter bins are the result; '
                                 'select only other filters.')
            mask = self._bin_mask(tally.filters[j], sel)
            values = np.compress(mask, values, axis=j + 1)
            shape[j] = int(mask.sum())
        values = np.moveaxis(values, i_root + 1, 1)
        values = values.reshape(self.n_batches, shape[i_root], -1).sum(axis=2)
        return DetectorResponse(
            values.reshape(self.detector_source.shape), self)

    @staticmethod
    def _filter_index(tally, key):
        """Position in the tally of a filter given as instance, class or ID."""
        for j, f in enumerate(tally.filters):
            if (f is key or (isinstance(key, type) and isinstance(f, key))
                    or (isinstance(key, int) and f.id == key)):
                return j
        raise ValueError(f'Tally {tally.id} has no filter matching {key!r}.')

    @staticmethod
    def _bin_mask(f, sel):
        """Boolean mask over a filter's bins from an index, indices or mask."""
        import openmc
        n = f.num_bins
        sel = np.asarray(sel)
        if sel.dtype == bool:
            if isinstance(f, openmc.MeshFilter) and sel.shape == tuple(
                    f.mesh.dimension):
                sel = sel.ravel(order='F')
            if sel.size != n:
                raise ValueError(f'A mask over filter {f.id} needs {n} '
                                 f'entries, not {sel.size}.')
            return sel.ravel()
        mask = np.zeros(n, dtype=bool)
        mask[np.atleast_1d(sel).astype(int)] = True
        return mask

    def depth_curve(self, quantity, **kwargs):
        """A quantity at every depth 1..n_generation, to check that the
        importance has converged by the last depth.

        Parameters
        ----------
        quantity : str
            Name of one of this object's estimator methods, e.g. 'beta_eff'
        **kwargs
            Passed to the method

        Returns
        -------
        list of uncertainties.UFloat

        """
        method = getattr(self, quantity)
        return [method(depth=d, **kwargs)
                for d in range(1, self.n_generation + 1)]


class DetectorResponse:
    """Detector response per unit source weight, per batch, by the category
    of the detector trees' roots and the mesh element of their origin.

    The detector trees are the non-multiplying response: the detector signal
    a neutron born at the origin causes before it is absorbed or leaks,
    fission counting as absorption. Folding it with a transient's sources,
    prompt fission neutrons and the delayed-neutron and photoneutron
    precursors' decay rates by origin, gives the signal split by what emitted
    the neutrons.

    Parameters
    ----------
    values : numpy.ndarray
        Response per unit source weight, [batch, category, origin]
    populations : openmc.AdjointPopulations
        The side populations it came from

    Attributes
    ----------
    values : numpy.ndarray
        Response per unit source weight, [batch, category, origin]
    source : numpy.ndarray
        Source weight per generation, [batch, category, origin]

    """

    def __init__(self, values, populations):
        self.values = np.asarray(values, float)
        self._pops = populations
        self.source = populations.detector_source
        off_pn, off_probe, off_ray, n_cat = populations.detector_categories
        self._off_pn, self._off_probe, self._off_ray = off_pn, off_probe, off_ray
        self._n_cat = n_cat

    @property
    def n_batches(self):
        return self.values.shape[0]

    @property
    def n_origins(self):
        return self.values.shape[2]

    def _mean(self, x):
        n = x.shape[0]
        m = x.mean(axis=0)
        s = x.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else \
            np.full_like(m, np.nan)
        return np.array([ufloat(a, b) for a, b in zip(m.ravel(), s.ravel())]
                        ).reshape(m.shape)

    def category(self, index):
        """Response per unit source weight of one category, per origin.

        Returns
        -------
        numpy.ndarray of uncertainties.UFloat

        """
        return self._mean(self.values[:, index, :])

    @property
    def prompt(self):
        """Response per prompt fission neutron, per origin"""
        return self.category(0)

    def delayed(self, group):
        """Response per delayed neutron of a group (1 to 8), per origin"""
        if not 1 <= group < self._off_pn:
            raise ValueError(f'No delayed group {group}.')
        return self.category(group)

    @property
    def photoneutrons(self):
        """Response per driver photoneutron, [photoneutron bin, origin]: one
        bin per energy group and fissioning-nuclide bin, and a last one for
        those outside the groups (a single bin without groups)"""
        return self._mean(self.values[:, self._off_pn:self._off_probe, :])

    def signal(self):
        """Steady-state detector signal per source particle by category,
        summed over the origins: per batch, response times source weight,
        normalized like the tallies. The probe and ray lines are not sources
        of their own; fold them with :meth:`photon_importance`.

        Returns
        -------
        dict of str to uncertainties.UFloat
            'prompt', 'delayed 1' to 'delayed 8' (those with a source),
            'photoneutrons' and 'total'

        """
        n = self._pops.n_particles
        if n is None:
            raise ValueError('The number of particles is unknown.')
        s = (self.values * self.source).sum(axis=2) / n
        parts = {'prompt': s[:, 0]}
        for g in range(1, self._off_pn):
            if self.source[:, g].any():
                parts[f'delayed {g}'] = s[:, g]
        if self._off_probe > self._off_pn:
            parts['photoneutrons'] = s[:, self._off_pn:self._off_probe].sum(
                axis=1)
        parts['total'] = sum(v for k, v in parts.items())
        return {k: self._mean(v[:, None])[0] for k, v in parts.items()}

    def photon_importance(self, target, nuclide=None, origin=0, rtol=0.01,
                          floor=1.0e-3, e_max=None, cross_sections=None):
        """Detector response to a photon born at a fission event, as a
        function of its energy, rebuilt from the probe (and ray) lines as
        :meth:`AdjointPopulations.probe_importance` rebuilds the fission
        importance.

        Its :meth:`openmc.ProbeImportance.fold` of a gamma source per fission
        is the detector response per unit fission weight (w/k
        sigma_f/sigma_t) of the nuclide bin at the origin, with batch
        statistics that keep the correlation of the lines.

        Parameters
        ----------
        target : openmc.PhotoneutronTarget or openmc.Material, or dict
            Cross sections of the medium that makes the photoneutrons, or,
            with the photoneutrons tallied by target, a dict of media as for
            :meth:`AdjointPopulations.probe_importance`
        nuclide : str, optional
            Fissioning-nuclide bin; may be omitted if there is only one
        origin : int
            Origin mesh element
        rtol, floor, e_max, cross_sections
            As for :meth:`AdjointPopulations.probe_importance`

        Returns
        -------
        openmc.ProbeImportance

        """
        from openmc.photoneutron_probes import ProbeImportance
        pops = self._pops
        pops._check_probes()
        target = pops._resolve_targets(target, cross_sections)
        j = pops._probe_bin(nuclide)
        K = len(pops.probe_energies)
        T = len(pops.probe_targets)
        v = self.values[:, :, origin]
        # [batch, line, target, label] and [batch, ray line, target]
        probe = v[:, self._off_probe:self._off_ray].reshape(
            self.n_batches, -1, K, T, 3)[:, j]
        rays = self._n_cat > self._off_ray
        if rays:
            KR = len(pops.ray_photon_energies)
            ray = v[:, self._off_ray:].reshape(
                self.n_batches, -1, KR, T)[:, j]
        ones = np.ones(self.n_batches)
        name = pops.probe_fission_nuclides[j]

        def parts(sel):
            p = probe[:, :, sel].sum(axis=2) if isinstance(sel, slice) \
                else probe[:, :, sel]
            r = (ray[:, :, sel].sum(axis=2) if isinstance(sel, slice)
                 else ray[:, :, sel]) if rays else None
            direct = p[:, :, 1] if rays else p[:, :, 0] + p[:, :, 1]
            return direct, p[:, :, 2], r

        if isinstance(target, dict):
            direct, scattered, ray_direct = {}, {}, {}
            for key in target:
                direct[key], scattered[key], ray_direct[key] = \
                    parts(pops.probe_targets.index(key))
        else:
            direct, scattered, ray_direct = parts(slice(None))
        return ProbeImportance(
            pops.probe_energies, direct, scattered, ones, target, rtol=rtol,
            floor=floor, e_max=e_max, name=name,
            ray_lines=pops.ray_photon_energies if rays else None,
            ray_direct=ray_direct if rays else None)


def _ratio(a, b):
    """mean(a)/mean(b) over batches with a delta-method sigma."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    ma, mb = a.mean(), b.mean()
    r = ma / mb
    if n < 2:
        return ufloat(r, np.nan)
    z = (a - ma) / mb - r * (b - mb) / mb
    return ufloat(r, z.std(ddof=1) / np.sqrt(n))


def _double_ratio(a, b, c, d):
    """(mean a / mean b) / (mean c / mean d) with a delta-method sigma."""
    x = np.stack([np.asarray(v, float) for v in (a, b, c, d)])
    m = x.mean(axis=1)
    r = (m[0] / m[1]) / (m[2] / m[3])
    n = x.shape[1]
    if n < 2:
        return ufloat(r, np.nan)
    g = np.array([1 / m[0], -1 / m[1], -1 / m[2], 1 / m[3]]) * r
    z = g @ (x - m[:, None])
    return ufloat(r, z.std(ddof=1) / np.sqrt(n))
