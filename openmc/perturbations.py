from __future__ import annotations

from numbers import Integral
from pathlib import Path

import lxml.etree as ET
import numpy as np

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
    rho : float
        Reactivity worth in pcm. Only present when read from a statepoint.
    std_dev : float
        One standard deviation of :attr:`rho`, in pcm. Only present when read
        from a statepoint. Use :meth:`Perturbations.difference` rather than
        combining these by hand: perturbations sharing branch sites are
        strongly correlated, and treating them as independent badly overstates
        the error on a difference.
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
        self.std_dev = None
        self.depth_curve = None

    def __repr__(self):
        parts = [f'LocalPerturbation\n{"":<12}ID={self.id}']
        if self.name:
            parts.append(f'{"":<12}Name={self.name}')
        parts.append(f'{"":<12}Substitutions={self.substitutions}')
        if self.rho is not None:
            parts.append(f'{"":<12}Worth={self.rho:.4g} +/- '
                         f'{self.std_dev:.4g} pcm')
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

    All perturbations are computed in one eigenvalue run. Those sharing a cell
    share branch sites and random seeds, so their worths come out strongly
    correlated and differences between them are far better determined than the
    individual values -- see :meth:`difference` and :meth:`combine`.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    perturbations : Iterable of openmc.LocalPerturbation
        Perturbations to add to the collection
    n_generation : int
        Number of shadow-tree generations, L. The worth is the slope of the
        log importance ratio over depth, so this must be at least 2; 10-12 is
        the usual starting point. Check :meth:`linearity` rather than
        assuming.

    Attributes
    ----------
    covariance : numpy.ndarray
        Covariance of the worths in pcm^2, ordered as the collection. Only
        present when read from a statepoint.

    """

    def __init__(self, perturbations=None, n_generation=10):
        super().__init__(LocalPerturbation, 'collection of perturbations')
        self.n_generation = n_generation
        self.covariance = None
        if perturbations is not None:
            self += perturbations

    @property
    def n_generation(self):
        return self._n_generation

    @n_generation.setter
    def n_generation(self, n):
        cv.check_type('n_generation', n, Integral)
        cv.check_greater_than('n_generation', n, 2, equality=True)
        self._n_generation = n

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
    def _set_results(self, numerators, denominators, n_blocks=None):
        """Derive worths and their covariance from recorded shadow weights.

        Parameters
        ----------
        numerators : numpy.ndarray
            ``tau_p(d)`` per generation, shape (n_perturbations, n_gen, L+1).
        denominators : numpy.ndarray
            The matched reference ``R_p(d)``, same shape.
        n_blocks : int, optional
            Number of groups the generations are split into for the
            delete-one-block jackknife.

        The worth is the slope of ``l_p(d) = ln[tau_p(d) / R_p(d)]``. The
        ratio is formed from sums over MANY generations, never one at a time:
        a shadow tree is a branching process that can go extinct, so a single
        generation's ``tau`` may be zero and ``log(0)`` is ``-inf``. Ordinary
        IFP estimators are robust to exactly this because they sum over every
        progenitor before dividing.
        """
        n_pert, n_gen, nd = numerators.shape
        L = nd - 1
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

        self._n_blocks = n_blocks
        self.covariance = 1.0e10 * cov
        for i, p in enumerate(self):
            p.rho = 1.0e5 * float(total[i])
            p.std_dev = 1.0e5 * float(np.sqrt(max(cov[i, i], 0.0)))
            with np.errstate(divide='ignore', invalid='ignore'):
                p.depth_curve = np.log(numerators[i].sum(0) /
                                       denominators[i].sum(0))

    def value(self, perturbation_id):
        """(rho, sigma) in pcm for one perturbation."""
        i = self.ids.index(perturbation_id)
        if self.covariance is None:
            raise ValueError('No results present; read from a statepoint.')
        return (float(self[i].rho),
                float(np.sqrt(max(self.covariance[i, i], 0.0))))

    @property
    def n_blocks(self):
        """Jackknife groups the generations were split into.

        The jackknife treats blocks as independent. Generations share a
        fission source, so uncertainties are somewhat optimistic; vary this
        and check the error bar is stable before relying on it.
        """
        return getattr(self, '_n_blocks', None)

    def _weights(self, weights):
        w = np.zeros(len(self))
        for key, value in weights.items():
            w[self.ids.index(key)] = value
        return w

    def combine(self, weights):
        """Value and uncertainty of a weighted sum of worths.

        Uses the full covariance, so a combination of strongly correlated
        perturbations gets the small uncertainty it deserves rather than the
        quadrature sum of the individual ones.

        Parameters
        ----------
        weights : dict
            Mapping of perturbation ID to weight

        Returns
        -------
        2-tuple of float
            Value and one standard deviation, in pcm

        """
        if self.covariance is None:
            raise ValueError('No results present; read from a statepoint.')
        w = self._weights(weights)
        rho = np.array([p.rho for p in self])
        return (float(w @ rho),
                float(np.sqrt(max(w @ self.covariance @ w, 0.0))))

    def difference(self, id_a, id_b):
        """``rho(a) - rho(b)`` with the correlation carried through."""
        return self.combine({id_a: 1.0, id_b: -1.0})

    def derivative(self, id_a, id_b, dz):
        """``[rho(a) - rho(b)] / dz``, in pcm per unit of ``dz``."""
        value, std_dev = self.difference(id_a, id_b)
        return value / dz, std_dev / abs(dz)

    def correlation(self):
        """Correlation matrix of the worths."""
        if self.covariance is None:
            raise ValueError('No results present; read from a statepoint.')
        d = np.sqrt(np.clip(np.diag(self.covariance), 1e-300, None))
        return self.covariance / np.outer(d, d)

    def linearity(self, perturbation_id, d_min=None):
        """Reduced chi-square of a straight-line fit to the depth curve.

        Near 1 means the asymptotic regime has been reached over the fitted
        range. Much above 1 means ``n_generation`` is too small and the
        estimator is contaminated by the transient. There is no plateau to
        look for: the curve is linear and its slope is the worth.

        """
        y = self.by_id(perturbation_id).depth_curve
        if y is None:
            raise ValueError('No results present; read from a statepoint.')
        d = np.arange(len(y))
        if d_min is None:
            d_min = self.n_generation // 2
        sel = d >= d_min
        if sel.sum() < 3:
            return float('nan')
        resid = y[sel] - np.polyval(np.polyfit(d[sel], y[sel], 1), d[sel])
        scale = np.abs(y[sel]).max() or 1.0
        return float((resid**2).sum() / (sel.sum() - 2)) / scale**2

    # ----------------------------------------------------------------- XML
    def to_xml_element(self):
        """Create a 'perturbations' element to be written to an XML file."""
        element = ET.Element('perturbations')
        ET.SubElement(element, 'n_generation').text = str(self.n_generation)
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
        n_gen = elem.find('n_generation')
        obj = cls(n_generation=10 if n_gen is None else int(n_gen.text))
        for sub in elem.findall('local_perturbation'):
            obj.append(LocalPerturbation.from_xml_element(sub))
        return obj

    @classmethod
    def from_xml(cls, path='perturbations.xml'):
        """Generate perturbations from a perturbations.xml file."""
        parser = ET.XMLParser(remove_comments=True)
        return cls.from_xml_element(ET.parse(str(path), parser=parser)
                                    .getroot())
