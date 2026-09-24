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
