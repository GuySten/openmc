import os
from numbers import Integral
from warnings import warn

import numpy as np
import h5py

import openmc.checkvalue as cv
from openmc.stats import Tabular
from . import HDF5_VERSION
from .ace import get_metadata, Table, Library
from .angle_distribution import AngleDistribution
from .data import ATOMIC_SYMBOL, EV_PER_MEV
from .energy_distribution import ContinuousTabular
from .function import Tabulated1D
from .photon import (_BREMSSTRAHLUNG, _SUBSHELLS, MASS_ELECTRON_EV,
                     _load_bremsstrahlung)

# Bremsstrahlung has no threshold-free cross section -- dsigma/dk diverges as
# 1/k -- so a lowest emitted photon energy in [eV] has to be chosen. At 1 eV the
# rate is comparable to the evaluated tables' and what is dropped below it is
# negligible.
_PHOTON_CUTOFF = 1.0


def _tabular_from_cdf(x, c, name):
    """Build a histogram Tabular from a tabulated cumulative distribution.

    ACE stores these distributions as abscissae plus cumulative probabilities.
    Converting to the histogram probabilities that :class:`Tabular` expects
    requires dividing by the bin widths, so a repeated abscissa would yield inf
    or nan and propagate silently into the exported library.

    """
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        bad = int(np.argmax(dx <= 0.0))
        raise ValueError(
            f'Non-increasing abscissa in {name}: value {x[bad]} at index {bad} '
            f'is followed by {x[bad + 1]}. The cumulative distribution cannot '
            'be differentiated.')
    p = np.append(np.diff(c) / dx, 0.0)
    dist = Tabular(x, p, interpolation='histogram')
    dist.c = c
    return dist


# Dirac partial-wave elastic cross sections are read from a pre-generated HDF5
# file when they are first needed. The dictionary stores the incident energies
# with the key 'energy' and the deflections 1-cos(theta) at which the cross
# sections are tabulated with the key 'mu'; the data for each element is a dict
# with the single key 'dcs' (a 2D array with shape (n_energies, n_angles),
# exponentiated on read from the logarithm the file stores), stored on the key
# Z. The differential cross section is all the file holds: every cross section
# the transport needs is an integral of it, and taking them from the same table
# that is sampled is what keeps the rate consistent with the deflections it
# produces.
_ELASTIC_DPWA = {}

def _with_cdf(x, p):
    """Tabular carrying the cumulative the HDF5 writers and the transport expect.

    AngleDistribution and ContinuousTabular both write a tabulated CDF
    alongside the density, and the sampler inverts that CDF rather than
    reintegrating the density. Building it here with the same trapezoidal rule
    the interpolation implies keeps the two consistent.
    """
    c = np.concatenate(([0.0], np.cumsum(0.5 * (p[:-1] + p[1:]) * np.diff(x))))
    total = c[-1]
    if total <= 0.0:
        raise ValueError('distribution has no probability')
    dist = Tabular(x, p / total, interpolation='linear-linear')
    dist.c = c / total
    return dist


def _log_interp(x, xp, fp):
    """Log-log interpolation, clamped to the endpoints of the tabulated range."""
    x = np.clip(x, xp[0], xp[-1])
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(fp)))


class IncidentElectron:
    """Continuous-energy incident electron interaction data parsed from ACE."""

    def __init__(self, atomic_number):
        self.atomic_number = atomic_number
        self.energy_grid = None
        self.elastic_xs = None
        self.elastic_dist = None
        self.bremsstrahlung_xs = None
        self.bremsstrahlung_photon_cutoff = None
        self.excitation_xs = None
        self.excitation_energy_loss = None        
        self.ionization_xs = {}  # Keyed by subshell index
        self.ionization_dist = {} # Keyed by subshell index
        self.shells = []

    def __repr__(self):
        return f"<IncidentElectron: {self.name}>"

    @property
    def atomic_number(self):
        return self._atomic_number

    @atomic_number.setter
    def atomic_number(self, atomic_number):
        cv.check_type("atomic number", atomic_number, Integral)
        cv.check_greater_than("atomic number", atomic_number, 0, True)
        self._atomic_number = atomic_number

    @property
    def name(self):
        return ATOMIC_SYMBOL[self.atomic_number]

    @classmethod
    def from_ace(cls, ace_table_or_filename, photon_cutoff=_PHOTON_CUTOFF):
        """Generate incident electron data from an ACE table

        The excitation and electroionization data come from the table. Elastic
        scattering and bremsstrahlung do not: their evaluated distributions are
        tabulated on far too few incident energies to interpolate, so they are
        taken from the calculated datasets distributed with OpenMC, the way the
        photon data takes its Compton profiles and its scaled bremsstrahlung
        cross sections.

        Parameters
        ----------
        ace_or_filename : str or openmc.data.ace.Table
            ACE table to read from. If given as a string, it is assumed to be
            the filename for the ACE file.
        photon_cutoff : float
            Lowest emitted bremsstrahlung photon energy in [eV]; see
            :meth:`IncidentElectron._add_bremsstrahlung`.

        Returns
        -------
        openmc.data.IncidentElectron
            Electron interaction data

        """
        if isinstance(ace_table_or_filename, Table):
            ace = ace_table_or_filename
        else:
            lib = Library(ace_table_or_filename)
            ace = lib.tables[0]

        # Initialize instance using ZA or Atomic Number
        Z = get_metadata(int(ace.zaid))[2]
        data = cls(Z)

        # Parse NXS/JXS array layout
        n_energy = ace.nxs[8]
        n_xl = ace.nxs[9]
        n_subshells = ace.nxs[7]
        

        j_shell = ace.jxs[11]           # SUBSH: subshell designators
        j_energy = ace.jxs[19]          # ESZE: electron energy grid + cross sections
        j_excitation = ace.jxs[20]      # EXCIT: excitation energy-loss table
        j_ionization = ace.jxs[23]      # EION: electroionization table info
        
        data.shells = [_SUBSHELLS[int(i)] for i in ace.xss[j_shell : j_shell + n_subshells]]
        data.energy_grid = ace.xss[j_energy : j_energy + n_energy]*EV_PER_MEV
        
        j_xs = j_energy + n_energy

        # Read cross sections from the ESZE block. The layout is, in order:
        # energy grid, total, elastic, bremsstrahlung, excitation, total
        # electroionization, then one block per subshell. Only excitation is
        # taken from it. The total and the total electroionization are
        # redundant, the first with the sum of the partials and the second
        # with the subshell blocks read below; the elastic and bremsstrahlung
        # columns belong to distributions this reader does not produce, and
        # reading them would only leave two numbers to be overwritten.
        data.excitation_xs = ace.xss[j_xs + 3 * n_energy : j_xs + 4 * n_energy]

        # Average excitation energy loss, from the EXCIT block at JXS(20).
        # This is NOT at JXS(5), which locates the photon heating numbers.
        # Both the abscissa and the tabulated loss are stored in MeV.
        data.excitation_energy_loss = Tabulated1D(
            ace.xss[j_excitation : j_excitation + n_xl]*EV_PER_MEV,
            ace.xss[j_excitation + n_xl : j_excitation + 2 * n_xl]*EV_PER_MEV)

        j_subshell_xs = j_xs + 5 * n_energy
        for s, shell in enumerate(data.shells):
            start_idx = j_subshell_xs + s * n_energy
            data.ionization_xs[shell] = ace.xss[start_idx : start_idx + n_energy]
            
        ni = ace.xss[j_ionization : j_ionization + n_subshells].astype(int)
        locinfo = ace.xss[j_ionization + n_subshells: j_ionization + 2*n_subshells].astype(int)
        loctab = ace.xss[j_ionization + 2*n_subshells: j_ionization + 3*n_subshells].astype(int)
        for s, shell in enumerate(data.shells):
            energy = ace.xss[locinfo[s]:locinfo[s]+ni[s]]*EV_PER_MEV
            ls = ace.xss[locinfo[s]+ni[s]:locinfo[s]+2*ni[s]].astype(int)
            offsets = ace.xss[locinfo[s]+2*ni[s]:locinfo[s]+3*ni[s]].astype(int)
            energy_out = []
            for i in range(ni[s]):
                start = loctab[s] + offsets[i]
                e = ace.xss[start:start + ls[i]]*EV_PER_MEV
                c = ace.xss[start + ls[i]:start + 2*ls[i]]
                energy_out.append(_tabular_from_cdf(
                    e, c, f'electroionization table {i} of subshell {shell}'))
            # Log-log interpolation between the tabulated incident energies.
            # These grids are extremely sparse -- aluminium's K shell jumps
            # from 15.8 keV to 501 keV with nothing in between -- and a linear
            # weight puts 83% of that interval on the lower table.
            #
            # Measured against the ICRU-37 collision stopping power between 50
            # and 300 keV, where the density effect vanishes, a linear weight
            # runs 5-9% low and a logarithmic one 1-3% low, consistently for
            # every element tested from beryllium to tantalum. The residual is
            # the shell correction, which the Bethe form of the reference
            # omits and which grows with Z in the same way.
            #
            # Unit-base scaling must stay off here, as the transport code has
            # it. These spectra are not self-similar -- the low end is anchored
            # near the binding energy while the tip follows the kinematic limit
            # (E - B)/2 -- so rescaling a low-energy table's shape onto a much
            # wider range drives the mean energy transfer far too high: with
            # unit-base on, the collision stopping power comes out at 1.3 to
            # 2.1 times ICRU-37.
            data.ionization_dist[shell] = ContinuousTabular(
                [len(energy)], [5], energy, energy_out)

        # Add partial-wave elastic and Seltzer-Berger bremsstrahlung data
        data._add_dpwa_elastic()
        data._add_bremsstrahlung(photon_cutoff)

        return data

    def export_to_hdf5(self, path, mode="a", libver="earliest"):
        """Export incident electron data to an HDF5 file.

        Parameters
        ----------
        path : str
            Path to write HDF5 file to
        mode : {'r+', 'w', 'x', 'a'}
            Mode that is used to open the HDF5 file. This is the second argument
            to the :class:`h5py.File` constructor.
        libver : {'earliest', 'latest'}
            Compatibility mode for the HDF5 file. 'latest' will produce files
            that are less backwards compatible but have performance benefits.

        """
        with h5py.File(str(path), mode, libver=libver) as f:
            # Write filetype and version
            f.attrs["filetype"] = np.bytes_("data_electron")
            if "version" not in f.attrs:
                f.attrs["version"] = np.array(HDF5_VERSION)

            group = f.create_group(self.name)
            group.attrs["Z"] = Z = self.atomic_number

            group.create_dataset("energy", data=self.energy_grid)
            
            elastic_group = group.create_group("elastic")
            elastic_group.create_dataset("xs", data=self.elastic_xs)
            self.elastic_dist.to_hdf5(elastic_group.create_group("distribution"))
            
            excitation_group = group.create_group("excitation")
            excitation_group.create_dataset("xs", data=self.excitation_xs)
            self.excitation_energy_loss.to_hdf5(excitation_group, "energy_loss")
            
            ionization_group = group.create_group("ionization")
            ionization_group.attrs['designators'] = np.array(self.shells, dtype='S')
            xs = np.zeros((len(self.shells), len(self.energy_grid)))
            for i, shell in enumerate(self.shells):
                xs[i] = self.ionization_xs[shell]
            ionization_group.create_dataset("xs", data=xs)
            
            for shell in self.shells:
                shell_group = ionization_group.create_group(shell)
                self.ionization_dist[shell].to_hdf5(shell_group)
            
            bremsstrahlung_group = group.create_group("bremsstrahlung")
            bremsstrahlung_group.create_dataset("xs", data=self.bremsstrahlung_xs)
            # The emitted photon energy is sampled from the scaled cross
            # sections of the photon library, so only the threshold this cross
            # section was integrated above is written -- the transport has to
            # sample above the same one or the rate and the spectrum stop being
            # integrals of the same thing.
            bremsstrahlung_group.attrs["photon_cutoff"] = \
                self.bremsstrahlung_photon_cutoff

    def _add_dpwa_elastic(self):
        """Add the Dirac partial-wave elastic scattering data.

        Read from ``elastic_dpwa.h5`` beside this module, the way the photon
        data finds its Compton profiles and its scaled bremsstrahlung cross
        sections.

        References
        ----------
        The data are computed with ELSEPA. If you use them in your research, please
        cite Salvat, Jablonski and Powell, *Computer Physics Communications* **165**
        (2005) 157-190.

        Notes
        -----
        The evaluated libraries split elastic scattering at mu = 1 - 1e-6, giving a
        tabulated distribution above that and leaving the forward peak to an
        analytic screened-Rutherford form, with the transport cross section as the
        only anchor tying the two together. A partial-wave differential cross
        section covers the whole angular range at once, so there is no split, no
        separate total to reconcile with it, and nothing for the peak-sampling and
        rescaling machinery in the transport to do. The cross section written here
        is the integral of the same distribution the transport samples, so the rate
        at which collisions happen is consistent with the deflections they give.

        The partial-wave data stops at 100 MeV where the evaluated data runs to
        100 GeV. Cross sections are clamped to the endpoints beyond that range,
        which is adequate below 100 MeV and wrong above it.

        """
        Z = self.atomic_number

        # Load the partial-wave elastic data if it has not yet been loaded
        if not _ELASTIC_DPWA:
            path = os.path.join(os.path.dirname(__file__), 'elastic_dpwa.h5')
            with h5py.File(path, 'r') as f:
                if f.attrs.get('filetype') != np.bytes_('elastic_dpwa'):
                    raise ValueError(f'{path} is not an elastic_dpwa file')
                _ELASTIC_DPWA['energy'] = f['energy'][()]
                # 1 - cos(theta), ascending from 0
                _ELASTIC_DPWA['mu'] = f['mu'][()]
                for i in range(1, 101):
                    key = f'{i:03}'
                    if key not in f:
                        continue
                    group = f[key]
                    _ELASTIC_DPWA[i] = {
                        # Stored as its logarithm; see make_elastic_dpwa.py
                        'dcs': np.exp(group['log_dcs'][()].astype(float))}

        energy = _ELASTIC_DPWA['energy']
        deflection = _ELASTIC_DPWA['mu']
        if Z not in _ELASTIC_DPWA:
            raise ValueError(f'No partial-wave elastic data for Z={Z}')
        dcs = _ELASTIC_DPWA[Z]['dcs']

        # The cross section is the integral of the distribution that is sampled,
        # not a separately tabulated number: 2*pi*int dcs d(1-cos(theta)). Taking
        # it from the same 606-point table keeps the rate at which collisions
        # happen consistent with the deflections they produce. It runs 0.2-1.2%
        # above ELSEPA's own phase-shift total, which is the quadrature error of
        # the tabulated grid and belongs in the rate as well.
        xs = 2.0 * np.pi * np.trapezoid(dcs, deflection, axis=1)

        grid = self.energy_grid
        if grid[0] < energy[0] or grid[-1] > energy[-1]:
            warn(f'{self.name}: the partial-wave data covers '
                 f'{energy[0]:.4g} to {energy[-1]:.4g} eV but the energy grid runs '
                 f'{grid[0]:.4g} to {grid[-1]:.4g} eV. Elastic cross sections are '
                 'clamped to the endpoints outside that range, which is wrong '
                 'rather than merely approximate.')
        barns = 1.0e24
        self.elastic_xs = _log_interp(grid, energy, xs) * barns

        # mu ascending from -1, as the transport samples it. The partial-wave cross
        # section is a density, so it is tabulated as one rather than differentiated
        # from a cumulative -- which is where the evaluated tables lose 1-2% of the
        # first moment to histogram binning.
        mu = (1.0 - deflection)[::-1]
        distributions = []
        for i in range(len(energy)):
            distributions.append(_with_cdf(mu, dcs[i][::-1]))
        self.elastic_dist = AngleDistribution(energy, distributions)


    def _add_bremsstrahlung(self, photon_cutoff=_PHOTON_CUTOFF):
        """Add the Seltzer-Berger bremsstrahlung cross section.

        Parameters
        ----------
        photon_cutoff : float
            Lowest emitted photon energy in [eV] to integrate the cross section
            above; see ``_PHOTON_CUTOFF``.

        Notes
        -----
        Only the cross section is written. The spectrum it integrates is the scaled
        cross section chi(Z, T, kappa) of the photon library, which the transport
        samples directly, so storing a copy of it here would be storing the same
        numbers twice and inviting the two to drift apart.

        The evaluated spectra are tabulated on nine incident energies for carbon,
        with nothing between 12.25 MeV and 100 GeV -- a factor of 8163. Seltzer and
        Berger give 57, sixteen of them between 0.256 and 25 MeV, so the anchoring
        against BREML that the sparse tables needed is not written here at all.

        """
        _load_bremsstrahlung()
        Z = self.atomic_number
        energy = _BREMSSTRAHLUNG['electron_energy']
        kappa = _BREMSSTRAHLUNG['photon_energy']
        chi = _BREMSSTRAHLUNG[Z]['dcs']

        grid = self.energy_grid
        if grid[0] < energy[0] or grid[-1] > energy[-1]:
            warn(f'{self.name}: the scaled bremsstrahlung cross sections cover '
                 f'{energy[0]:.4g} to {energy[-1]:.4g} eV but the energy grid runs '
                 f'{grid[0]:.4g} to {grid[-1]:.4g} eV. Cross sections are clamped '
                 'to the endpoints outside that range.')

        # sigma = int_{k_cut}^{T} dsigma/dk dk with dsigma/dk = Z^2/beta^2 chi/k,
        #       = Z^2/beta^2 int_{kappa_cut}^{1} chi(kappa)/kappa dkappa.
        # chi is linear in kappa between tabulated points, which the transport
        # assumes when it samples, so each interval integrates in closed form:
        # int (a + b kappa)/kappa dkappa = a ln(kappa2/kappa1) + b (kappa2-kappa1).
        # Doing it analytically rather than on a quadrature grid is what keeps the
        # rate an integral of exactly the distribution that is sampled.
        gamma = 1.0 + energy/MASS_ELECTRON_EV
        beta_sq = 1.0 - 1.0/(gamma*gamma)
        kappa_cut = photon_cutoff/energy

        lo, hi = kappa[:-1], kappa[1:]
        b = np.diff(chi, axis=1)/(hi - lo)
        a = chi[:, :-1] - b*lo
        # Clip each interval to [kappa_cut, 1]; intervals below the cutoff collapse
        x1 = np.clip(lo, kappa_cut[:, None], None)
        x2 = np.clip(hi, kappa_cut[:, None], None)
        integral = np.sum(
            np.where(x2 > x1, a*np.log(np.where(x2 > x1, x2/np.maximum(x1, 1e-300),
                                                1.0)) + b*(x2 - x1), 0.0),
            axis=1)
        xs = Z*Z/beta_sq*integral

        self.bremsstrahlung_xs = _log_interp(grid, energy, xs)
        self.bremsstrahlung_photon_cutoff = photon_cutoff
