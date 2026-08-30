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

        Perturbations sharing branch sites are strongly correlated, so a
        difference formed this way has a far smaller uncertainty than the
        quadrature sum of the two individual ones.
    std_dev : float
        Shorthand for ``rho.std_dev``, in pcm.
    depth_curve : numpy.ndarray
        ``l(d)``, the log importance ratio against shadow-tree depth, whose
        slope is :attr:`rho`. Only present when read from a statepoint.

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

    def __repr__(self):
        parts = [f'LocalPerturbation\n{"":<12}ID={self.id}']
        if self.name:
            parts.append(f'{"":<12}Name={self.name}')
        parts.append(f'{"":<12}Substitutions={self.substitutions}')
        if self.rho is not None:
            parts.append(f'{"":<12}Worth={self.rho:.4g} pcm')
        return '\n'.join(parts) + '\n'

    @property
    def std_dev(self):
        """One standard deviation of :attr:`rho`, in pcm."""
        return None if self.rho is None else self.rho.std_dev

    @property
    def nominal_value(self):
        """Central value of :attr:`rho`, in pcm."""
        return None if self.rho is None else self.rho.nominal_value

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

    All perturbations are computed in one eigenvalue run. Those sharing a cell
    share branch sites and random seeds, so their worths come out strongly
    correlated and differences between them are far better determined than the
    individual values. Each :attr:`LocalPerturbation.rho` is a correlated
    :mod:`uncertainties` value, so that is automatic::

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
        depths the recorded curves span. To CHOOSE it, set
        :attr:`openmc.Settings.perturbation_n_generation` -- it applies to
        the whole run, not to one perturbation, since every shadow tree is
        compared against the same reference trees at the same depths.

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
    def _set_results(self, numerators, denominators, k_ref=1.0,
                     n_blocks=None):
        """Derive worths and their covariance from recorded shadow weights.

        Parameters
        ----------
        numerators : numpy.ndarray
            ``tau_p(d)`` per generation, shape (n_perturbations, n_gen, L+1).
        denominators : numpy.ndarray
            The matched reference ``R_p(d)``, same shape.
        k_ref : float
            k-effective of the reference system, used to convert the fitted
            slope into a reactivity. Defaults to 1, which leaves the result
            in dk/k.
        n_blocks : int, optional
            Number of groups the generations are split into for the
            delete-one-block jackknife.

        The ratio is formed from sums over MANY generations, never one at a
        time: a shadow tree is a branching process that can go extinct, so a
        single generation's ``tau`` may be zero and ``log(0)`` is ``-inf``.
        Ordinary IFP estimators are robust to exactly this because they sum
        over every progenitor before dividing.

        The slope of ``l_p(d) = ln[tau_p(d) / R_p(d)]`` is
        ``ln(k_p / k_ref)``, i.e. dk/k -- NOT a reactivity. The reactivity
        difference is ``1/k_ref - 1/k_p``, which is smaller by a factor of k.
        Reporting the slope directly would overstate every worth by that
        factor, so it is converted here, exactly rather than to first order::

            rho = (1 - exp(-slope)) / k_ref

        k_ref carries its own uncertainty, but as a common multiplicative
        factor of order 1e-5 relative it is negligible against the worths and
        is not propagated.
        """
        n_pert, n_gen, nd = numerators.shape
        L = nd - 1
        # The results describe their own depth, so a collection read back
        # from a statepoint is self-contained even if the caller never sees
        # the Settings that produced it.
        self._n_generation = L
        d_min = L // 2
        d = np.arange(nd)
        fit = d >= d_min
        if fit.sum() < 2:
            raise ValueError('n_generation is too small to fit a slope')

        def slope(num, den):
            """Least-squares slope of ln(num/den) over the fitted depths."""
            x = d[fit] - d[fit].mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                ell = np.log(num[..., fit] / den[..., fit])
                return (ell * x).sum(-1) / (x**2).sum()

        # Point estimate from the whole run: the largest possible sums, so
        # extinction of individual trees is irrelevant.
        total = slope(numerators.sum(1), denominators.sum(1))

        # Uncertainty and covariance by delete-one-block jackknife. The
        # estimator is a slope of a log of a ratio, so it is nonlinear in the
        # accumulated sums and the spread of independent per-block estimates
        # is the wrong thing to average. Each jackknife replicate instead uses
        # all but one block, so it is as well-conditioned as the full estimate
        # -- extinction inside a single block cannot make a replicate
        # degenerate, which plain blocking could not promise.
        if n_blocks is None:
            n_blocks = min(20, n_gen)
        n_blocks = max(2, min(n_blocks, n_gen))
        edges = np.linspace(0, n_gen, n_blocks + 1).astype(int)

        num_all, den_all = numerators.sum(1), denominators.sum(1)
        num_drop = np.array([num_all - numerators[:, a:b].sum(1)
                             for a, b in zip(edges[:-1], edges[1:])])
        den_drop = np.array([den_all - denominators[:, a:b].sum(1)
                             for a, b in zip(edges[:-1], edges[1:])])
        if (numerators[..., 1:] == 0).all():
            raise DataError(
                'Every shadow tree has zero weight beyond depth 0, so no '
                'tree ever produced a fission site. That is a build or '
                'configuration fault, not a statistics one: check that the '
                'fission-site creation gate in sample_neutron_reaction() and '
                'the revival gate in event_check_limit_and_revive() both use '
                'bep::generation_limit(), rather than '
                'settings::super_n_generation directly.')

        if not ((num_drop > 0).all() and (den_drop > 0).all()):
            raise DataError(
                'A jackknife replicate has a shadow tree that is extinct at '
                'every depth, so no uncertainty can be formed. Increase the '
                'particles per generation so more branch sites are recorded, '
                'enlarge the perturbed region, or reduce n_generation.')

        replicates = slope(num_drop, den_drop)        # (n_blocks, n_pert)
        centred = replicates - replicates.mean(0)
        cov = (n_blocks - 1) / n_blocks * (centred.T @ centred)
        cov = np.atleast_2d(cov)

        # Keep the depth curves, per replicate as well as in total. Every
        # honest diagnostic about the fit -- whether the residuals are
        # consistent with the noise, whether the slope has stopped moving as
        # the fit window shrinks -- needs their covariance, and the jackknife
        # replicates are the only source of it.
        with np.errstate(divide='ignore', invalid='ignore'):
            self._ell = np.log(num_all / den_all)             # (n_pert, nd)
            self._ell_replicates = np.log(num_drop / den_drop)
        self._k_ref = k_ref
        self._fit_from = d_min

        # Slope (dk/k) -> reactivity. The Jacobian of
        # rho = (1 - exp(-s))/k_ref is exp(-s)/k_ref, essentially 1/k_ref for
        # any realistic worth, and the covariance transforms with it.
        jacobian = np.exp(-total) / k_ref
        total = (1.0 - np.exp(-total)) / k_ref
        cov = cov * np.outer(jacobian, jacobian)

        self._n_blocks = n_blocks

        # Hand the whole covariance to uncertainties rather than storing a
        # scalar sigma per perturbation. Every rho then carries its
        # correlations with the others, so a difference, a derivative or any
        # weighted combination propagates correctly with no bookkeeping here
        # and none in user code.
        rho = correlated_values(1.0e5 * total, 1.0e10 * cov)
        for i, p in enumerate(self):
            p.rho = rho[i]
            with np.errstate(divide='ignore', invalid='ignore'):
                p.depth_curve = np.log(numerators[i].sum(0) /
                                       denominators[i].sum(0))

    @property
    def n_blocks(self):
        """Jackknife groups the generations were split into.

        The jackknife treats blocks as independent. Generations share a
        fission source, so uncertainties are somewhat optimistic; vary this
        and check the error bar is stable before relying on it.
        """
        return getattr(self, '_n_blocks', None)

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

    def _replicate_stats(self, values):
        """Jackknife mean and standard deviation of a per-replicate array."""
        n = values.shape[0]
        centred = values - values.mean(0)
        return values.mean(0), np.sqrt((n - 1) / n * (centred**2).sum(0))

    def linearity(self, perturbation_id, d_min=None):
        """Reduced chi-square of the straight-line fit to the depth curve.

        A real chi-square: residuals are weighted by the jackknife covariance
        of the depth curve, so the scale is meaningful. Near 1 means the
        deviations from a straight line are no larger than the noise, i.e. the
        asymptotic regime has been reached over the fitted range. Well above 1
        means it has not, and the fitted worth is biased -- see
        :meth:`worth_by_fit_start`, which shows that bias directly.

        The points are strongly correlated across depth, so the covariance is
        used in full rather than point-by-point error bars.
        """
        if getattr(self, '_ell', None) is None:
            raise ValueError('No results present; read from a statepoint.')
        i = self.ids.index(perturbation_id)
        d = np.arange(self._ell.shape[1])
        sel = d >= (self._fit_from if d_min is None else d_min)
        if sel.sum() < 3:
            return float('nan')

        y = self._ell[i, sel]
        reps = self._ell_replicates[:, i, :][:, sel]
        n = reps.shape[0]
        centred = reps - reps.mean(0)
        cov = (n - 1) / n * (centred.T @ centred)

        resid = y - np.polyval(np.polyfit(d[sel], y, 1), d[sel])
        # pinv, not inv: with n_blocks replicates the covariance has rank at
        # most n_blocks - 1 and can be ill-conditioned.
        chi2 = float(resid @ np.linalg.pinv(cov, rcond=1e-10) @ resid)
        return chi2 / (sel.sum() - 2)

    def worth_by_fit_start(self, perturbation_id):
        """Worth in pcm as a function of where the slope fit starts.

        The one diagnostic that answers "has the transient died out?"
        directly. A sub-dominant mode decaying as (dominance ratio)**d biases
        the slope, always in the same direction, and the bias shrinks as the
        fit starts later. If these values drift with the starting depth, the
        quoted worth is biased and ``n_generation`` is too small; if they
        plateau, it is not.

        Returns
        -------
        dict
            Starting depth -> (worth, sigma) in pcm, fitted over that depth
            through L.
        """
        if getattr(self, '_ell', None) is None:
            raise ValueError('No results present; read from a statepoint.')
        i = self.ids.index(perturbation_id)
        nd = self._ell.shape[1]
        d = np.arange(nd)

        def fit(curve, d0):
            sel = d >= d0
            x = d[sel] - d[sel].mean()
            return ((curve[..., sel] - curve[..., sel].mean(-1, keepdims=True))
                    * x).sum(-1) / (x**2).sum()

        out = {}
        for d0 in range(nd - 2):
            total = float(fit(self._ell[i], d0))
            reps = fit(self._ell_replicates[:, i, :], d0)
            _, sd = self._replicate_stats(reps[:, None])
            jac = np.exp(-total) / self._k_ref
            out[d0] = (1.0e5 * (1.0 - np.exp(-total)) / self._k_ref,
                       1.0e5 * float(sd[0]) * jac)
        return out

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
