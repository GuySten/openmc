from __future__ import annotations

from numbers import Integral
from pathlib import Path

import lxml.etree as ET
import numpy as np

import uncertainties
from uncertainties import correlated_values

import openmc
import openmc.checkvalue as cv
from openmc.exceptions import DataError
from ._xml import clean_indentation, get_text
from .mixin import IDManagerMixin

__all__ = ['LocalPerturbation', 'Perturbations']


class LocalPerturbation(IDManagerMixin):
    """A set of cell-material substitutions whose reactivity worth is computed.

    Every substitution in a perturbation is applied together, so a single
    :class:`LocalPerturbation` covers a sample swap (one cell), a rigid sample
    displacement (the trailing sliver reverting and the leading sliver taking
    the sample), or a multi-region change such as a voided plenum.

    The geometry must hold the *reference* material in every substituted cell;
    the substitution is applied only inside this perturbation's own shadow
    tree, leaving the driver calculation bit-identical to a stock run.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    substitutions : dict or iterable of 2-tuple
        Mapping of cell ID (or :class:`openmc.Cell`) to the material to
        substitute there, given as a material ID, an :class:`openmc.Material`,
        or None for void.
    perturbation_id : int, optional
        Unique identifier. Assigned automatically if not given.
    name : str, optional
        Name of the perturbation.

    Attributes
    ----------
    id : int
        Unique identifier
    name : str
        Name of the perturbation
    substitutions : dict
        Mapping of cell ID to material ID
    rho : uncertainties.UFloat
        Reactivity worth in pcm, or None until read from a statepoint.

        This is a correlated quantity, built with
        :func:`uncertainties.correlated_values` from the full covariance of
        the run, so arithmetic between perturbations propagates correctly with
        no further bookkeeping::

            a, b = sp.perturbations
            b.rho - a.rho          # difference, correlation carried through
            (b.rho - a.rho) / dz   # derivative, pcm per unit dz
            0.5 * (a.rho + b.rho)  # any combination you like

        Perturbations substituting into the same cell emit the same
        nuclides' source from the same seeds, so they are strongly
        correlated and a difference formed this way has a far smaller
        uncertainty than the quadrature sum of the two individual ones.
    depth_curve : numpy.ndarray
        The level against shadow-tree depth, in pcm: the same estimator
        :attr:`rho` is, read at every depth ``d = 0..L`` instead of only at
        ``L``. It is FLAT once the perturbed fundamental mode has established
        itself, so its approach to a plateau is the convergence diagnostic --
        there is no fit window here and so no window bias to correct. Only
        present when read from a statepoint.
    depth_sigma : numpy.ndarray
        Batch-statistics standard deviation of :attr:`depth_curve`, pcm.
    rho_pooled : float
        The worth in pcm formed from the run's POOLED totals rather than as a
        mean of per-batch levels. A ratio of means has no ratio-of-means bias
        and this one does not; the two agreeing to a small fraction of
        :attr:`rho`'s sigma is the single assumption the batch statistics
        make, and it can be checked directly. Quote this one if they ever
        disagree.

    """

    next_id = 1
    used_ids = set()

    def __init__(self, substitutions=None, perturbation_id=None, name=''):
        self.id = perturbation_id
        self.name = name
        self.substitutions = {} if substitutions is None else substitutions

        # Populated only when read from a statepoint
        self.rho = None
        self.depth_curve = None
        self.depth_sigma = None
        self.rho_pooled = None

    def __repr__(self):
        parts = [f'LocalPerturbation\n{"":<12}ID={self.id}']
        if self.name:
            parts.append(f'{"":<12}Name={self.name}')
        parts.append(f'{"":<12}Substitutions={self.substitutions}')
        if self.rho is not None:
            parts.append(f'{"":<12}Worth={self.rho:.4g} pcm')
        return '\n'.join(parts) + '\n'

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            cv.check_type('perturbation name', name, str)
            self._name = name
        else:
            self._name = ''

    @property
    def substitutions(self):
        return self._substitutions

    @substitutions.setter
    def substitutions(self, substitutions):
        if not isinstance(substitutions, dict):
            substitutions = dict(substitutions)

        subs = {}
        for cell, material in substitutions.items():
            cell_id = cell.id if isinstance(cell, openmc.Cell) else cell
            cv.check_type('perturbation cell', cell_id, Integral)
            if material is None:
                mat_id = 0
            elif isinstance(material, openmc.Material):
                mat_id = material.id
            else:
                cv.check_type('perturbation material', material, Integral)
                mat_id = material
            subs[cell_id] = mat_id
        self._substitutions = subs

    @property
    def cells(self):
        """IDs of the cells this perturbation touches."""
        return list(self._substitutions.keys())

    def to_xml_element(self):
        """Return an XML representation of the perturbation.

        Returns
        -------
        lxml.etree._Element
            ``<local_perturbation>`` element

        """
        elem = ET.Element('local_perturbation')
        elem.set('id', str(self.id))
        if self.name:
            elem.set('name', self.name)
        for cell_id, mat_id in self._substitutions.items():
            sub = ET.SubElement(elem, 'substitution')
            ET.SubElement(sub, 'cell').text = str(cell_id)
            ET.SubElement(sub, 'material').text = str(mat_id)
        return elem

    @classmethod
    def from_xml_element(cls, elem):
        """Generate a perturbation from an XML element.

        Parameters
        ----------
        elem : lxml.etree._Element
            ``<local_perturbation>`` element

        Returns
        -------
        openmc.LocalPerturbation

        """
        subs = {}
        for sub in elem.findall('substitution'):
            subs[int(get_text(sub, 'cell'))] = int(get_text(sub, 'material'))
        # Bare <cell>/<material> shorthand for the one-cell case
        if elem.find('cell') is not None:
            subs[int(get_text(elem, 'cell'))] = int(get_text(elem, 'material'))
        return cls(subs, perturbation_id=int(elem.get('id')),
                   name=elem.get('name', ''))


class Perturbations(cv.CheckedList):
    """Collection of local perturbations used for an OpenMC simulation.

    This class corresponds directly to the perturbations.xml input file. It can
    be thought of as a normal Python list where each member is a
    :class:`LocalPerturbation`, and is assigned to
    :attr:`openmc.Model.perturbations`:

    >>> model.perturbations = openmc.Perturbations([
    ...     openmc.LocalPerturbation({sample_cell: steel}),
    ...     openmc.LocalPerturbation({sample_cell: zircaloy}),
    ... ])

    All perturbations are computed in one eigenvalue run. They share the
    driver, the fission source and -- where they touch the same cell -- their
    random seeds, so their worths come out strongly correlated and differences
    between them are far better determined than the individual values. Each
    :attr:`LocalPerturbation.rho` is a correlated :mod:`uncertainties` value,
    so that is automatic::

        a, b = sp.perturbations
        b.rho - a.rho            # correlation carried through
        (b.rho - a.rho) / dz     # pcm per unit dz

    .. versionadded:: 0.16.0

    Parameters
    ----------
    perturbations : Iterable of openmc.LocalPerturbation
        Perturbations to add to the collection

    Attributes
    ----------
    n_generation : int
        Shadow tree depth, L. Read-only here, and only meaningful on a
        collection read back from a statepoint, where it says how many
        depths the recorded curves span. The worth is the level at ``d = L``.
        To CHOOSE it, set
        :attr:`openmc.Settings.perturbation_n_generation` -- it applies to
        the whole run, not to one perturbation.
    n_batches : int
        Active batches behind the batch statistics.

    """

    def __init__(self, perturbations=None):
        super().__init__(LocalPerturbation, 'collection of perturbations')
        self._n_generation = None
        if perturbations is not None:
            self += perturbations

    @property
    def n_generation(self):
        """Shadow tree depth of the run these results came from."""
        return self._n_generation

    @property
    def ids(self):
        return [p.id for p in self]

    def __getitem__(self, key):
        """Index by position, or by ID when the key is not a valid position."""
        if isinstance(key, (int, np.integer)) and key in self.ids \
                and not (0 <= key < len(self)):
            return self[self.ids.index(key)]
        return super().__getitem__(key)

    def by_id(self, perturbation_id):
        """Return the perturbation with the given ID."""
        return super().__getitem__(self.ids.index(perturbation_id))

    # ------------------------------------------------------------- results
    def _set_results(self, level_sum, level_cross, level_pooled, n_batches,
                     tau_pooled=None, trees=None, keff=None):
        """Derive worths and their covariance from the level batch statistics.

        Parameters
        ----------
        level_sum : numpy.ndarray
            Sum over active batches of the per-batch level, shape
            (n_perturbations, L+1). Already a reactivity (1/k - 1/k'), in
            absolute units -- the C++ does the whole estimator, including the
            exact inversion for k', because it is the same three lines
            whatever the analysis layer wants to do afterwards.
        level_cross : numpy.ndarray
            Sum over active batches of the outer product of the per-batch
            levels, shape (n_perturbations, n_perturbations, L+1).
        level_pooled : numpy.ndarray
            The same level formed from the run's pooled totals instead of as a
            mean of per-batch values, shape (n_perturbations, L+1).
        n_batches : int
            Active batches behind the sums: the realization count of the batch
            statistics, exactly as for any other OpenMC tally.
        tau_pooled : numpy.ndarray, optional
            The five populations' run totals, shape (n_trees, L+1). Needed by
            :meth:`amplification` and :meth:`convergence`; without it those
            raise.
        trees : list of numpy.ndarray, optional
            Each perturbation's five tree indices into ``tau_pooled``, in the
            C++'s class order D, F+, F-, L+, L-.
        keff : float, optional
            The eigenvalue the fission bank was normalised by.

        The uncertainty is the ordinary spread of the per-batch realizations,
        with no model of the correlation between generations anywhere. That is
        what ``generations_per_batch`` is for: raise it until sigma stops
        growing, and the batches are independent enough for this to be honest.
        Check that plateau once per problem; it is the only thing standing
        between these error bars and the truth.
        """
        level_sum = np.asarray(level_sum, dtype=float)
        level_cross = np.asarray(level_cross, dtype=float)
        level_pooled = np.asarray(level_pooled, dtype=float)
        n_pert, nd = level_sum.shape
        L = nd - 1
        # The results describe their own depth, so a collection read back
        # from a statepoint is self-contained even if the caller never sees
        # the Settings that produced it.
        self._n_generation = L
        self._n_batches = int(n_batches)

        if self._n_batches < 2:
            raise DataError(
                'Batch statistics need at least two active batches; this run '
                'has {}. Increase batches, or reduce '
                'generations_per_batch.'.format(self._n_batches))

        n = float(self._n_batches)
        mean = level_sum / n

        # Covariance OF THE MEAN, in one step from the accumulated sums:
        #     cov_ij = (S_ij - n * mean_i * mean_j) / (n * (n - 1))
        # The off-diagonal is not decoration. Two perturbations in one run
        # share the driver, the fission source and -- where they touch the
        # same cell -- their seeds, so most of their noise is common; a
        # difference between them is far better determined than the
        # quadrature sum of the two sigmas would suggest.
        outer = mean[:, None, :] * mean[None, :, :]
        cov = (level_cross - n * outer) / (n * (n - 1.0))

        # Numerical noise can leave a tiny negative variance where a
        # perturbation is identically zero (the null test), and
        # correlated_values will not take that.
        for i in range(n_pert):
            if cov[i, i, L] < 0.0:
                cov[i, i, L] = 0.0

        self._mean = mean
        self._cov = cov
        self._pooled = level_pooled
        self._tau = None if tau_pooled is None else np.asarray(
            tau_pooled, dtype=float)
        self._trees = trees
        self._keff = keff

        # pcm, and pcm^2 for the covariance.
        rho = correlated_values(1.0e5 * mean[:, L], 1.0e10 * cov[:, :, L])
        sigma = np.sqrt(np.abs(np.einsum('iid->id', cov)))
        for i, p in enumerate(self):
            p.rho = rho[i]
            p.depth_curve = 1.0e5 * mean[i]
            p.depth_sigma = 1.0e5 * sigma[i]
            p.rho_pooled = 1.0e5 * float(level_pooled[i, L])

    @property
    def n_batches(self):
        """Active batches behind the batch statistics, or None."""
        return getattr(self, '_n_batches', None)

    @property
    def covariance(self):
        """Covariance of the worths in pcm^2, ordered as the collection.

        Recovered from the correlated :attr:`LocalPerturbation.rho` values, so
        it always agrees with what their arithmetic produces. None until read
        from a statepoint.
        """
        if len(self) == 0 or any(p.rho is None for p in self):
            return None
        return np.array(uncertainties.covariance_matrix([p.rho for p in self]))

    def correlation(self):
        """Correlation matrix of the worths.

        A perturbation with zero variance has no correlation with anything --
        a null perturbation is exactly that, being identically zero by
        construction -- so its row and column come back ``nan`` rather than
        the zero that would falsely read as "uncorrelated".

        Computed from :attr:`covariance` rather than delegating to
        :func:`uncertainties.correlation_matrix`, which divides by the
        standard deviations unguarded and so raises a ``RuntimeWarning`` on
        the same input.
        """
        cov = self.covariance
        if cov is None:
            raise ValueError('No results present; read from a statepoint.')
        std_dev = np.sqrt(np.diag(cov))
        with np.errstate(divide='ignore', invalid='ignore'):
            return cov / np.outer(std_dev, std_dev)

    def confidence_interval(self, perturbation_id, level=0.95):
        """Two-sided t-interval on the worth, in pcm.

        Student's t on ``n_batches - 1`` degrees of freedom, not a normal
        quantile: with the twenty or so batches a perturbation run typically
        has, the difference is several per cent of the interval and always in
        the direction of under-stating it.

        Returns
        -------
        tuple of float
            Lower and upper bound in pcm.
        """
        from scipy.stats import t as student_t

        if getattr(self, '_mean', None) is None:
            raise ValueError('No results present; read from a statepoint.')
        i = self.ids.index(perturbation_id)
        rho = self[i].rho
        half = student_t.ppf(0.5 * (1.0 + level),
                             self._n_batches - 1) * rho.std_dev
        return (rho.nominal_value - half, rho.nominal_value + half)

    def pooled_vs_batch(self, perturbation_id):
        """How far the pooled worth sits from the mean of the batch levels.

        A mean of ratios is not the ratio of means, and the difference is the
        one bias batch statistics can introduce here. This returns it in units
        of the worth's own sigma: anything much below 0.1 is negligible, and
        anything approaching 1 means :attr:`LocalPerturbation.rho_pooled` is
        the number to quote.
        """
        if getattr(self, '_mean', None) is None:
            raise ValueError('No results present; read from a statepoint.')
        i = self.ids.index(perturbation_id)
        p = self[i]
        if p.rho.std_dev == 0.0:
            return 0.0
        return (p.rho_pooled - p.rho.nominal_value) / p.rho.std_dev

    def amplification(self, perturbation_id):
        """How much the estimator magnifies a transient, per depth.

        The worth is a small difference of large populations, twice over: the
        fission channel against the removal channel, and within the removal
        channel the positive part against the negative one. A transient that
        is one part in a thousand of the populations is still a large
        fraction of the ANSWER, and it is the answer it has to be small
        against.

        This returns, per depth, the ratio of the magnitudes being
        subtracted to the magnitude of what survives::

            amplification(d) = (|N_F|/k + |N_L| + |L+| + |L-|) / |N_F/k + N_L|

        Measured 440 on a fuel-density perturbation, where the level ran from
        +19 to −154 pcm over twelve generations against a true worth of
        +1.2. It is the number that decides how deep ``L`` has to be; see
        :meth:`convergence`.

        Returns
        -------
        numpy.ndarray
            Amplification at each depth ``0..L``.
        """
        if self._tau is None:
            raise ValueError('No population totals present; this statepoint '
                             'predates tau_pooled, or was built by hand.')
        i = self.ids.index(perturbation_id)
        d_w, fp, fn, lp, ln = self._tau[self._trees[i]]
        k = self._keff
        n_f, n_l = fp - fn, lp - ln
        num = np.abs(n_f) / k + np.abs(n_l) + lp + ln
        den = np.abs(n_f / k + n_l)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(den > 0.0, num / den, np.inf)

    def convergence(self, perturbation_id, target=0.05, d_min=None):
        """Is the level converged, and if not how deep would it have to be?

        Fits ``A + B r**d`` to the level curve. ``r`` is the rate the
        transient dies at -- the dominance ratio of the perturbed system, as
        the estimator sees it -- ``A`` is the extrapolated asymptote, and
        ``B r**L`` is what is still left to go at the depth actually run.

        **This replaces judging flatness by eye, and it replaces comparing
        the drift to the noise.** Comparing to the noise is what makes an
        unconverged level look converged: at forty replicas a 6.5 pcm drift
        sat comfortably inside a 12 pcm tolerance and passed, while the worth
        it licensed was 6.5% wrong. The drift has to be compared to the
        ANSWER, magnified by :meth:`amplification`.

        Parameters
        ----------
        target : float
            Fractional precision wanted on the worth.
        d_min : int, optional
            First depth to fit from. Defaults to 1, since depth 0 is the
            source counted with no propagation at all and is not on the curve.

        Returns
        -------
        dict
            ``asymptote`` (pcm), ``remaining`` (pcm still to go at ``L``),
            ``rate`` (the fitted ``r``), ``amplification`` at ``L``,
            ``required_depth`` for ``target``, and ``converged``.
        """
        if getattr(self, '_mean', None) is None:
            raise ValueError('No results present; read from a statepoint.')
        i = self.ids.index(perturbation_id)
        p = self[i]
        curve = np.asarray(p.depth_curve, dtype=float)
        L = len(curve) - 1
        d = np.arange(len(curve), dtype=float)
        sel = d >= (1 if d_min is None else d_min)
        x, y = d[sel], curve[sel]

        best = None
        for r in np.linspace(0.05, 0.999, 2000):
            M = np.vstack([np.ones_like(x), r ** x]).T
            try:
                coef, *_ = np.linalg.lstsq(M, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            chi2 = float(((y - M @ coef) ** 2).sum())
            if best is None or chi2 < best[0]:
                best = (chi2, r, coef[0], coef[1])
        _, r, A, B = best
        remaining = B * r ** L

        amp = np.inf
        try:
            amp = float(self.amplification(perturbation_id)[L])
        except ValueError:
            pass
        with np.errstate(divide='ignore', invalid='ignore'):
            need = (np.log(amp / target) / np.log(1.0 / r)
                    if 0.0 < r < 1.0 and np.isfinite(amp) else np.inf)
        return {
            'asymptote': float(A),
            'remaining': float(remaining),
            'rate': float(r),
            'amplification': amp,
            'required_depth': float(need),
            'converged': bool(abs(remaining) <= target * abs(A) and L >= need),
        }

    def depth_convergence(self, perturbation_id):
        """Level against depth for one perturbation, as a diagnostic.

        Returns
        -------
        dict
            Depth -> (level, sigma) in pcm. The level is flat in depth once
            the perturbed fundamental mode has established itself; if it is
            still moving at ``d = L`` then ``perturbation_n_generation`` is
            too small and the reported worth has not converged.

            This is the curve to look at, but do not judge it by eye and do
            not compare its drift to its error bars -- use
            :meth:`convergence`, which compares the drift to the answer
            magnified by :meth:`amplification`. A curve whose drift is small
            against the noise can still carry a worth that is badly wrong.
        """
        if getattr(self, '_mean', None) is None:
            raise ValueError('No results present; read from a statepoint.')
        i = self.ids.index(perturbation_id)
        p = self[i]
        return {d: (float(p.depth_curve[d]), float(p.depth_sigma[d]))
                for d in range(len(p.depth_curve))}

    # ----------------------------------------------------------------- XML
    def to_xml_element(self):
        """Create a 'perturbations' element to be written to an XML file."""
        # No n_generation here: it lives in settings.xml, parsed before this
        # file is read, so anything that needs it early can see it.
        element = ET.Element('perturbations')
        for perturbation in self:
            element.append(perturbation.to_xml_element())
        clean_indentation(element)
        return element

    def export_to_xml(self, path='perturbations.xml'):
        """Create a perturbations.xml file for a simulation.

        Parameters
        ----------
        path : str
            Path to file to write. Defaults to 'perturbations.xml'.

        """
        p = Path(path)
        if p.is_dir():
            p /= 'perturbations.xml'
        ET.ElementTree(self.to_xml_element()).write(
            str(p), xml_declaration=True, encoding='utf-8')

    @classmethod
    def from_xml_element(cls, elem):
        """Generate perturbations from an XML element."""
        obj = cls()
        for sub in elem.findall('local_perturbation'):
            obj.append(LocalPerturbation.from_xml_element(sub))
        return obj

    @classmethod
    def from_xml(cls, path='perturbations.xml'):
        """Generate perturbations from a perturbations.xml file."""
        parser = ET.XMLParser(remove_comments=True)
        return cls.from_xml_element(ET.parse(str(path), parser=parser)
                                    .getroot())
