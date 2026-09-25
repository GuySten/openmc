"""Photon importance from probe photons of the adjoint side populations.

With ``adjoint_populations['photoneutron_probe_energies']`` set, the side
populations emit one probe photon at every recorded fission event, at one of a
comb of line energies E_k, with unit intensity per line per fission. The
importance of its photoneutrons is tallied per fissioning nuclide and line,
split by whether the probe had changed energy before making them:

- direct, I_d(E): photoneutrons made at the line energy. Their production
  follows the target's photoneutron cross section sigma(E), resonances and
  all, and the photon's chance to reach its first energy-changing collision,
  1/Sigma'(E);
- scattered, I_s(E): photoneutrons made after a Compton scattering (or by a
  secondary photon). Scattering spreads the energy over a wide range, so I_s
  is smooth in E, apart from the chance that the first collision is a
  scattering at all, again 1/Sigma'(E).

Here sigma(E) is the target's macroscopic photoneutron production cross
section and Sigma'(E) its total photon cross section less coherent scattering
(which keeps the energy and so keeps a photon direct). The smooth functions

    A(E) = I_d(E) Sigma'(E) / sigma(E),    B(E) = I_s(E) Sigma'(E)

are interpolated linearly in photoneutron lethargy, ln(E - E_thr), between the
lines, and the importance of a photon of any energy is rebuilt with the exact
cross sections:

    I(E) = [sigma(E) A(E) + B(E)] / Sigma'(E).

A photon line of intensity Y per fission then adds Y I(E) to the photoneutron
reactivity. :class:`ProbeImportance` holds I(E) for one fissioning nuclide,
tabulated on a grid refined until linear interpolation of the table is within
a given tolerance of the rebuilt function.

"""

from collections.abc import Callable, Iterable
from numbers import Real
import os

import numpy as np
from uncertainties import ufloat

import openmc.checkvalue as cv


def probe_line_energies(threshold, e_max, n, first=1.0e3):
    """Probe line energies spaced evenly in photoneutron lethargy.

    Parameters
    ----------
    threshold : float
        Photoneutron threshold of the target [eV]
    e_max : float
        Highest line [eV]
    n : int
        Number of lines
    first : float
        Energy of the lowest line above the threshold [eV]

    Returns
    -------
    numpy.ndarray
        Line energies [eV], threshold + geomspace(first, e_max - threshold, n)

    """
    cv.check_greater_than('e_max', e_max, threshold + first)
    cv.check_greater_than('n', n, 1)
    return threshold + np.geomspace(first, e_max - threshold, n)


class PhotoneutronTarget:
    """Photoneutron production and removal cross sections of a target medium.

    Parameters
    ----------
    energy : Iterable of float
        Energy grid [eV] on which both cross sections are exact under linear
        interpolation (their union of breakpoints)
    sigma : Iterable of float
        Macroscopic photoneutron production cross section on the grid [1/cm]:
        sum over reactions and neutron products of sigma_rx times the yield
    sigma_removal : Iterable of float
        Macroscopic total photon cross section less coherent scattering, on
        the grid [1/cm]

    Attributes
    ----------
    energy, sigma, sigma_removal : numpy.ndarray
    threshold : float
        Photoneutron threshold [eV]: the last grid point with zero production
        before the first with nonzero production

    """

    def __init__(self, energy, sigma, sigma_removal):
        e = np.asarray(energy, float)
        s = np.asarray(sigma, float)
        r = np.asarray(sigma_removal, float)
        if e.ndim != 1 or s.shape != e.shape or r.shape != e.shape:
            raise ValueError('energy, sigma and sigma_removal must be 1-D '
                             'arrays of one length.')
        if np.any(np.diff(e) <= 0.0):
            raise ValueError('energy must be strictly increasing.')
        if not np.any(s > 0.0):
            raise ValueError('sigma is zero everywhere.')
        if np.any(r[s > 0.0] <= 0.0):
            raise ValueError('sigma_removal must be positive where sigma is.')
        self.energy, self.sigma, self.sigma_removal = e, s, r
        i = int(np.argmax(s > 0.0))
        self.threshold = float(e[i - 1] if i > 0 else e[0])

    def sigma_at(self, E):
        return np.interp(E, self.energy, self.sigma, left=0.0, right=0.0)

    def sigma_removal_at(self, E):
        return np.interp(E, self.energy, self.sigma_removal)

    @classmethod
    def from_material(cls, material, cross_sections=None, e_max=20.0e6):
        """Cross sections of a material from the photon and photonuclear data
        of a cross_sections.xml, as OpenMC forms them in transport.

        Parameters
        ----------
        material : openmc.Material
            The target medium, e.g. the heavy water or beryllium
        cross_sections : str or os.PathLike, optional
            cross_sections.xml; openmc.config['cross_sections'] if None
        e_max : float
            Highest energy of the grid [eV]

        Returns
        -------
        PhotoneutronTarget

        """
        import openmc.data
        from openmc.data.data import ATOMIC_SYMBOL, zam

        if cross_sections is None:
            cross_sections = openmc.config.get('cross_sections')
        if cross_sections is None:
            raise ValueError('No cross_sections.xml given or configured.')
        library = openmc.data.DataLibrary.from_xml(cross_sections)
        densities = material.get_nuclide_atom_densities()

        elements = {}
        for name, n in densities.items():
            sym = ATOMIC_SYMBOL[zam(name)[0]]
            elements[sym] = elements.get(sym, 0.0) + n

        removal, production, grid = [], [], []
        for sym, n in elements.items():
            entry = library.get_by_material(sym, data_type='photon')
            if entry is None:
                raise ValueError(f'No photon data for {sym} in {cross_sections}.')
            data = openmc.data.IncidentPhoton.from_hdf5(entry['path'])
            for mt in (504, 515, 517, 522):
                if mt in data.reactions:
                    xs = data.reactions[mt].xs
                    removal.append((n, xs))
                    if hasattr(xs, 'x'):
                        grid.append(np.asarray(xs.x))
        for name, n in densities.items():
            entry = library.get_by_material(name, data_type='photonuclear')
            if entry is None:
                continue
            data = openmc.data.IncidentPhotonuclear.from_hdf5(entry['path'])
            for rx in data.reactions.values():
                xs = rx.xs
                if hasattr(xs, 'x'):
                    grid.append(np.asarray(xs.x))
                if not rx.redundant:
                    removal.append((n, xs))
                for prod in rx.products:
                    if prod.particle == 'neutron':
                        production.append((n, xs, prod.yield_))
        if not production:
            raise ValueError(f'Material {material.name!r} has no photoneutron '
                             'production in the photonuclear data.')

        e = np.unique(np.concatenate(grid))
        e = e[(e > 0.0) & (e <= e_max)]
        sig = np.zeros_like(e)
        for n, xs, y in production:
            yield_ = np.asarray(y(e), float)
            sig += n * np.where(yield_ > 0.0, xs(e) * yield_, 0.0)
        rem = np.zeros_like(e)
        for n, xs in removal:
            rem += n * np.asarray(xs(e), float)
        return cls(e, sig, rem)


class ProbeImportance:
    """Importance of a photon born at a fission of one nuclide, as a function
    of its energy, rebuilt from the probe lines.

    The importance is the photoneutron reactivity (per unit fission-neutron
    importance, like :meth:`AdjointPopulations.photoneutron_reactivity`) that
    a photon line of unit intensity per fission of the nuclide would add.

    Parameters
    ----------
    lines : numpy.ndarray
        Probe line energies [eV]
    direct, scattered : numpy.ndarray
        Summed probe-root weight per batch and line, [batch, line]: the
        photoneutrons made at the line energy and after an energy-changing
        collision. With ray_lines, direct holds only the probes' part (after
        coherent scattering) and the rays' part is given separately.
    fission_importance : numpy.ndarray
        Summed fission-root weight per batch at the same depth, [batch]
    target : PhotoneutronTarget
        Cross sections of the photoneutron target
    rtol : float
        Tolerance of linear interpolation of the refined table (the table is
        refined to half of it at 20 points per interval, for a margin)
    floor : float
        Fraction of the largest importance below which the tolerance is
        absolute (floor times that maximum)
    e_max : float, optional
        Top of the table [eV]; the highest line if None
    name : str, optional
        Name of the fissioning-nuclide bin
    ray_lines : numpy.ndarray, optional
        The rays' line energies [eV], if the rays were evaluated on lines of
        their own
    ray_direct : numpy.ndarray, optional
        The rays' uncollided photoneutron weight per batch and ray line,
        [batch, ray line]; interpolated like the direct part, on ray_lines

    Attributes
    ----------
    energy : numpy.ndarray
        Refined energy grid [eV]
    mean, std_dev : numpy.ndarray
        Importance on the grid and its batch standard deviation
    rtol, floor : float
    name : str

    """

    def __init__(self, lines, direct, scattered, fission_importance, target,
                 rtol=0.01, floor=1.0e-3, e_max=None, name=None,
                 ray_lines=None, ray_direct=None):
        self.lines = np.asarray(lines, float)
        self._fission = np.asarray(fission_importance, float)
        self.target = target
        self.rtol = float(rtol)
        self.floor = float(floor)
        self.name = name
        self.ray_lines = None if ray_lines is None else \
            np.asarray(ray_lines, float)
        grids = [self.lines] + ([] if ray_lines is None else [self.ray_lines])
        if min(g[0] for g in grids) <= target.threshold:
            raise ValueError('Every probe line must lie above the target '
                             'threshold.')
        self.e_max = float(max(g[-1] for g in grids) if e_max is None
                           else e_max)

        # The parts, each on its own lines, as node values per batch of a
        # smooth function: 'A' = I Sigma'/sigma for the photoneutrons made at
        # the line energy, 'B' = I Sigma' for those made after an
        # energy-changing collision. A line where sigma vanishes carries no
        # 'A' part.
        self._parts = []
        self._add_part('A', self.lines, direct)
        self._add_part('B', self.lines, scattered)
        if ray_lines is not None:
            self._add_part('A', self.ray_lines, ray_direct)

        self.energy = self._refine()
        self.mean, self.std_dev = self._statistics(self.energy)

    # ------------------------------------------------------------------------
    # The rebuilt function

    def _add_part(self, kind, lines, data):
        t = self.target
        s = t.sigma_at(lines)
        r = t.sigma_removal_at(lines)
        if kind == 'A':
            with np.errstate(divide='ignore', invalid='ignore'):
                scale = np.where(s > 0.0, r / s, 0.0)
        else:
            scale = r
        self._parts.append((kind, lines, np.log(lines - t.threshold), scale,
                            np.atleast_2d(np.asarray(data, float))))

    def _coefficients(self, E):
        """One matrix C per part such that the importance at E, per batch,
        is sum(C @ I_part) / I_F."""
        E = np.atleast_1d(np.asarray(E, float))
        thr = self.target.threshold
        above = E > thr
        u = np.log(np.where(above, E - thr, 1.0))
        s = self.target.sigma_at(E)
        r = self.target.sigma_removal_at(E)
        inv = np.where(above & (r > 0.0), 1.0 / np.where(r > 0.0, r, 1.0), 0.0)
        rows = np.arange(E.size)
        out = []
        for kind, lines, lu, scale, _ in self._parts:
            # Linear interpolation weights in lethargy, held flat above the
            # last line; below the first line A is held and B falls linearly
            # in E to zero at the threshold
            K = lines.size
            j = np.clip(np.searchsorted(lu, u) - 1, 0, K - 2)
            t = np.clip((u - lu[j]) / (lu[j + 1] - lu[j]), 0.0, 1.0)
            W = np.zeros((E.size, K))
            W[rows, j] = 1.0 - t
            W[rows, j + 1] += t
            if kind == 'A':
                out.append((s * inv)[:, None] * W * scale[None, :])
            else:
                low = above & (E < lines[0])
                if np.any(low):
                    W[low] = 0.0
                    W[low, 0] = (E[low] - thr) / (lines[0] - thr)
                out.append(inv[:, None] * W * scale[None, :])
        return out

    def _batch_numerators(self, E):
        C = self._coefficients(E)
        return sum(part[4] @ c.T for part, c in zip(self._parts, C))

    def rebuilt(self, E):
        """The rebuilt importance at energies E (batch means), exact in the
        cross sections; the refined table interpolates it."""
        num = self._batch_numerators(E).mean(axis=0)
        return num / self._fission.mean()

    def _statistics(self, E):
        num = self._batch_numerators(E)
        f = self._fission
        mf = f.mean()
        m = num.mean(axis=0) / mf
        n = f.size
        if n < 2:
            return m, np.full_like(m, np.nan)
        z = (num - num.mean(axis=0)) / mf - m[None, :] * (f - mf)[:, None] / mf
        return m, z.std(axis=0, ddof=1) / np.sqrt(n)

    # ------------------------------------------------------------------------
    # The refined table

    def _tolerance(self, values, scale):
        return self.rtol * np.maximum(np.abs(values), self.floor * scale)

    def _refine(self, n_check=20, max_rounds=60):
        t = self.target
        lo = t.threshold
        hi = self.e_max
        seeds = [t.energy[(t.energy > lo) & (t.energy < hi)], [lo, hi]] + \
            [part[1] for part in self._parts]
        grid = np.unique(np.concatenate(seeds))
        grid = grid[(grid >= lo) & (grid <= hi)]
        scale = np.max(np.abs(self.rebuilt(grid)))
        fr = np.arange(1, n_check + 1) / (n_check + 1)
        for _ in range(max_rounds):
            a, b = grid[:-1], grid[1:]
            pts = a[:, None] + fr[None, :] * (b - a)[:, None]
            exact = self.rebuilt(pts.ravel()).reshape(pts.shape)
            ya, yb = self.rebuilt(a), self.rebuilt(b)
            lin = ya[:, None] + fr[None, :] * (yb - ya)[:, None]
            scale = max(scale, np.max(np.abs(exact)))
            # Refined to half the tolerance, so that the points between
            # those checked stay within it
            bad = np.any(np.abs(lin - exact) >
                         0.5 * self._tolerance(exact, scale), axis=1)
            # Stop splitting at the resolution of double precision
            bad &= (b - a) > 1e-9 * b
            if not np.any(bad):
                break
            grid = np.unique(np.concatenate([grid, 0.5 * (a[bad] + b[bad])]))
        self._scale = scale
        return grid

    def __call__(self, E):
        """Linear interpolation of the refined table."""
        return np.interp(E, self.energy, self.mean, left=0.0, right=0.0)

    def interpolation_error(self, n_points=20, energies=None):
        """Largest error of linear interpolation of the table against the
        rebuilt function, relative to max(|I|, floor * max I), at n_points in
        every interval and at the given energies.

        Returns
        -------
        float

        """
        a, b = self.energy[:-1], self.energy[1:]
        fr = np.arange(1, n_points + 1) / (n_points + 1)
        pts = (a[:, None] + fr[None, :] * (b - a)[:, None]).ravel()
        if energies is not None:
            e = np.asarray(energies, float)
            pts = np.concatenate([pts, e[(e > a[0]) & (e < b[-1])]])
        exact = self.rebuilt(pts)
        scale = max(self._scale, np.max(np.abs(exact)))
        if scale == 0.0:
            # An empty bin (no probe reached it): nothing to interpolate
            return 0.0
        den = np.maximum(np.abs(exact), self.floor * scale)
        return float(np.max(np.abs(self(pts) - exact) / den))

    # ------------------------------------------------------------------------
    # Folding a photon source

    def fold(self, line_energies=(), line_intensities=(), continuum=None):
        """Photoneutron reactivity of a photon source per fission of this
        nuclide, with the rebuilt function and batch statistics.

        Parameters
        ----------
        line_energies, line_intensities : Iterable of float
            Discrete lines [eV] and their intensities per fission
        continuum : tuple of numpy.ndarray, optional
            (energy [eV], intensity per fission per eV), integrated with the
            trapezoidal rule on the union of its grid and the target's

        Returns
        -------
        uncertainties.UFloat

        """
        n = self._fission.size
        num = np.zeros(n)
        e = np.asarray(line_energies, float)
        y = np.asarray(line_intensities, float)
        if e.size:
            num += self._batch_numerators(e) @ y
        if continuum is not None:
            ce, cp = (np.asarray(v, float) for v in continuum)
            t = self.target.energy
            x = np.unique(np.concatenate(
                [ce, t[(t >= ce[0]) & (t <= ce[-1])]]))
            x = x[x > self.target.threshold]
            if x.size > 1:
                p = np.interp(x, ce, cp, left=0.0, right=0.0)
                w = np.zeros_like(x)
                dx = np.diff(x)
                w[:-1] += 0.5 * dx
                w[1:] += 0.5 * dx
                num += self._batch_numerators(x) @ (w * p)
        f = self._fission
        mf, mn = f.mean(), num.mean()
        r = mn / mf
        if n < 2:
            return ufloat(r, np.nan)
        z = (num - mn) / mf - r * (f - mf) / mf
        return ufloat(r, z.std(ddof=1) / np.sqrt(n))

    def to_csv(self, path):
        """Write the refined table: energy [eV], importance, standard
        deviation."""
        header = 'energy_eV,importance,std_dev'
        np.savetxt(path, np.column_stack([self.energy, self.mean,
                                          self.std_dev]),
                   delimiter=',', header=header, comments='', fmt='%.10e')
