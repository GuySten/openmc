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
    probe_weight : numpy.ndarray or None
        Summed probe-root weight, indexed [batch, nuclide bin, line, label,
        depth]. With three labels: 0 uncollided (empty when rays were on),
        1 after coherent scattering only, 2 after an energy-changing
        collision. Files written before the labels have two: 0 at the line
        energy (uncollided or coherent), 1 after an energy-changing collision
    probe_n_labels : int or None
        Number of probe labels (2 or 3)
    ray_weight : numpy.ndarray or None
        Summed uncollided photoneutron-root weight of the ray probes,
        indexed [batch, nuclide bin, ray line, depth], if rays were on
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
    probe_trigger : tuple of float or None
        (threshold, floor) of the probe-importance trigger, if one was set

    """

    def __init__(self, group, keff):
        self.n_generation = int(group['n_generation'][()])
        self.n_batches = int(group['n_batches'][()])
        self.photoneutrons = bool(group['photoneutrons'][()])
        self.perturbed_importance = bool(group['perturbed_importance'][()]) \
            if 'perturbed_importance' in group else False
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
        self.probe_weight = None
        self.probe_fission_weight = None
        if 'probe_energies' in group:
            self.probe_energies = group['probe_energies'][()]
            names = group['probe_fission_nuclides'][()]
            names = names.decode() if isinstance(names, bytes) else str(names)
            self.probe_fission_nuclides = names.split()
            nn = len(self.probe_fission_nuclides)
            self.probe_n_labels = int(group['probe_n_labels'][()]) \
                if 'probe_n_labels' in group else 2
            self.probe_weight = group['probe_weight'][()].reshape(
                (self.n_batches, nn, len(self.probe_energies),
                 self.probe_n_labels, self.n_generation + 1))
            self.probe_fission_weight = \
                group['probe_fission_weight'][()].reshape(self.n_batches, nn)
        self.probe_trigger = tuple(float(x) for x in group['probe_trigger'][()]) \
            if 'probe_trigger' in group else None
        self.ray_weight = None
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
            self.ray_weight = group['ray_weight'][()].reshape(
                (self.n_batches, nn, len(self.ray_photon_energies), nd))
            self.ray_comb_weight = group['ray_comb_weight'][()].reshape(
                (self.n_batches, nn, len(self.ray_neutron_energies), nd))
            self.ray_fission_weight = \
                group['ray_fission_weight'][()].reshape(self.n_batches, nn)

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
        target : openmc.PhotoneutronTarget or openmc.Material
            Cross sections of the medium that makes the photoneutrons, or the
            material itself (its cross sections are then formed with
            :meth:`openmc.PhotoneutronTarget.from_material`)
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
        from openmc.photoneutron_probes import (ProbeImportance,
                                                 PhotoneutronTarget)
        if not isinstance(target, PhotoneutronTarget):
            target = PhotoneutronTarget.from_material(target, cross_sections)
        self._check_probes()
        j = self._probe_bin(nuclide)
        d = self._depth(depth)
        i_f = self.weight[:, CLASS_FISSION, :, d].sum(axis=1)
        name = self.probe_fission_nuclides[j]
        if self.ray_weight is not None and self.probe_n_labels == 3:
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
        from openmc.photoneutron_probes import PhotoneutronTarget
        self._check_probes()
        if not isinstance(target, PhotoneutronTarget):
            target = PhotoneutronTarget.from_material(target, cross_sections)
        return {n: self.probe_importance(target, nuclide=n, depth=depth,
                                         rtol=rtol, floor=floor, e_max=e_max)
                for n in self.probe_fission_nuclides}

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
