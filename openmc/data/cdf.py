"""Cumulative distributions as tabulated in nuclear data files.

ACE tables and OpenMC's HDF5 nuclear data library store, alongside each
tabulated probability density, the cumulative distribution evaluated at the
same abscissae. OpenMC preserves those values verbatim rather than
recomputing them from the density so that sampling reproduces ACE results --
see the disabled reconstruction in :file:`src/secondary_correlated.cpp` and
:file:`src/secondary_kalbach.cpp`. Some evaluations are also not normalized
(NJOY thermal data in particular), and the tabulated values are the only
record of that.

This module is the single definition of that convention. The classes here
are used by :mod:`openmc.data` when reading ACE and HDF5 files; the
distributions in :mod:`openmc.stats` deliberately know nothing about it.

"""

from collections.abc import Iterable
from numbers import Real
from warnings import warn

import numpy as np

import openmc.checkvalue as cv
from openmc.stats import Discrete, Tabular

__all__ = ['TabulatedCDFMixin', 'TabularCDF', 'DiscreteCDF', 'cdf_values']


class TabulatedCDFMixin:
    """Mixin for a distribution carrying a CDF tabulated in a data file.

    .. versionadded:: 0.16.0

    Attributes
    ----------
    tabulated_cdf : numpy.ndarray or None
        Cumulative distribution evaluated at each value of `x`, exactly as
        tabulated in the ACE or HDF5 file the distribution was read from, or
        None if the distribution did not come from a file. It is not
        necessarily normalized; some evaluations are not. A decreasing value
        warns rather than raising, so that data OpenMC reads today keeps
        working. When a distribution read from a file is split into the
        discrete and continuous parts of an :class:`openmc.stats.Mixture`, the
        two arrays are consecutive pieces of one joint CDF and the discrete
        part's last entry is the probability of the discrete component.

        Note that this is *not* the same as the :meth:`cdf` method, which
        always derives cumulative values from `p`. In particular, for a
        discrete distribution the file convention has one value per point,
        whereas :meth:`openmc.stats.Discrete.cdf` prepends a zero. Use
        :func:`cdf_values` to obtain values in the file convention for any
        distribution. Mutating `p` directly leaves `tabulated_cdf` stale;
        :meth:`normalize` and :meth:`prepend` keep the two consistent.

    """

    @property
    def tabulated_cdf(self):
        return self._tabulated_cdf

    @tabulated_cdf.setter
    def tabulated_cdf(self, tabulated_cdf):
        if tabulated_cdf is None:
            self._tabulated_cdf = None
            return

        cv.check_type('tabulated CDF', tabulated_cdf, Iterable, Real)
        c = np.asarray(tabulated_cdf, dtype=float)
        if c.size != self.x.size:
            raise ValueError(
                f'Tabulated CDF has {c.size} values but the distribution has '
                f'{self.x.size} tabulated values of the random variable.')

        # A decreasing cumulative distribution indicates bad data, but the
        # values are passed through rather than rejected so that libraries
        # which OpenMC reads today keep working
        if np.any(c[1:] < c[:-1]):
            warn('Tabulated CDF is not non-decreasing.')

        self._tabulated_cdf = c

    @property
    def c(self):
        warn('The "c" attribute is deprecated in favor of "tabulated_cdf". '
             'Use openmc.data.cdf_values() to get cumulative values in the '
             'convention used by ACE and HDF5 files.', FutureWarning)
        return self._tabulated_cdf

    @c.setter
    def c(self, c):
        warn('The "c" attribute is deprecated in favor of "tabulated_cdf".',
             FutureWarning)
        self.tabulated_cdf = c

    def normalize(self):
        """Normalize the probabilities stored on the distribution.

        The tabulated CDF, if present, is rescaled so that it still
        corresponds to the normalized probabilities.

        """
        c = self._tabulated_cdf
        super().normalize()
        if c is not None and c[-1] > 0.0:
            self._tabulated_cdf = c / c[-1]

    def prepend(self, x, p=0.0, c=0.0):
        """Insert a point at the beginning of the tabulated data.

        Parameters
        ----------
        x : float
            Value of the random variable to insert
        p : float
            Probability to insert
        c : float
            Cumulative value to insert. Ignored if the distribution carries no
            tabulated CDF.

        """
        cv.check_type('x value', x, Real)
        cv.check_type('probability', p, Real)
        self._x = np.insert(self.x, 0, x)
        self._p = np.insert(self.p, 0, p)
        if self._tabulated_cdf is not None:
            self._tabulated_cdf = np.insert(self._tabulated_cdf, 0, c)


class TabularCDF(TabulatedCDFMixin, Tabular):
    """Piecewise continuous distribution with a CDF tabulated in a data file.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    x : Iterable of float
        Tabulated values of the random variable
    p : Iterable of float
        Tabulated probabilities
    interpolation : {'histogram', 'linear-linear', 'linear-log', 'log-linear', 'log-log'}, optional
        Indicates how the density function is interpolated between tabulated
        points. Defaults to 'linear-linear'.
    ignore_negative : bool
        Ignore negative probabilities
    tabulated_cdf : Iterable of float, optional
        Cumulative distribution as tabulated in the originating data file

    """

    def __init__(self, x, p, interpolation='linear-linear',
                 ignore_negative=False, tabulated_cdf=None):
        super().__init__(x, p, interpolation, ignore_negative)
        self.tabulated_cdf = tabulated_cdf


class DiscreteCDF(TabulatedCDFMixin, Discrete):
    """Discrete distribution with a CDF tabulated in a data file.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    x : Iterable of float
        Values of the random variable
    p : Iterable of float
        Discrete probabilities
    tabulated_cdf : Iterable of float, optional
        Cumulative distribution as tabulated in the originating data file

    """

    def __init__(self, x, p, tabulated_cdf=None):
        super().__init__(x, p)
        self.tabulated_cdf = tabulated_cdf


def cdf_values(dist):
    """Cumulative values for a distribution in the data file convention.

    ACE tables and OpenMC's HDF5 nuclear data library tabulate a cumulative
    distribution alongside each probability density, with one value per
    tabulated value of the random variable. This function returns the values
    tabulated in the originating file when `dist` carries them, and otherwise
    computes them from the probability density.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    dist : openmc.stats.Tabular or openmc.stats.Discrete
        Distribution to get cumulative values for

    Returns
    -------
    numpy.ndarray
        Cumulative values, of the same length as ``dist.x``

    """
    c = getattr(dist, 'tabulated_cdf', None)

    if c is None:
        # Cumulative values used to be attached to distributions from
        # openmc.stats as a bare 'c' attribute. Honor that if we find it, but
        # read it out of __dict__ so that the deprecated property on
        # TabulatedCDFMixin isn't triggered.
        legacy = dist.__dict__.get('c')
        if legacy is not None:
            warn('Attaching cumulative values to a distribution as a "c" '
                 'attribute is deprecated. Use openmc.data.TabularCDF or '
                 'openmc.data.DiscreteCDF instead.', FutureWarning)
            c = np.asarray(legacy, dtype=float)

    if c is not None:
        return c

    if isinstance(dist, Discrete):
        # The file convention has one cumulative value per point, whereas
        # Discrete.cdf() prepends a zero
        return dist.cdf()[1:]
    elif isinstance(dist, Tabular):
        return dist.cdf()
    else:
        raise TypeError(
            'No cumulative distribution convention is defined for '
            f'{type(dist).__name__}.')
