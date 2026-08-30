from numbers import Integral

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
from .photon import _SUBSHELLS
from .uncorrelated import UncorrelatedAngleEnergy

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


class IncidentElectron:
    """Continuous-energy incident electron interaction data parsed from ACE."""

    def __init__(self, atomic_number):
        self.atomic_number = atomic_number
        self.energy_grid = None
        self.elastic_xs = None
        self.elastic_dist = None
        self.bremsstrahlung_xs = None
        self.bremsstrahlung_dist = None
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
    def from_ace(cls, ace_table_or_filename):
        """Generate incident electron data from an ACE table

        Parameters
        ----------
        ace_or_filename : str or openmc.data.ace.Table
            ACE table to read from. If given as a string, it is assumed to be
            the filename for the ACE file.

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

        # Check the format flag. NXS(6) == 1 is EPRDATA12; 3 is EPRDATA14 and
        # later. Nothing read below moved between those versions, but a future
        # format could shift these blocks and misparse silently.
        format_flag = ace.nxs[6]
        if format_flag not in (1, 3):
            raise ValueError(
                f'Unrecognized electron-photon-relaxation format flag '
                f'NXS(6)={format_flag} in table {ace.name}. Supported values '
                'are 1 (EPRDATA12) and 3 (EPRDATA14 and later).')

        # Parse NXS/JXS array layout
        n_energy = ace.nxs[8]
        n_xl = ace.nxs[9]
        n_subshells = ace.nxs[7]
        

        j_shell = ace.jxs[11]           # SUBSH: subshell designators
        j_energy = ace.jxs[19]          # ESZE: electron energy grid + cross sections
        j_excitation = ace.jxs[20]      # EXCIT: excitation energy-loss table
        j_elastic = ace.jxs[21]         # ELASI: elastic angular table info
        j_elastic_tab = ace.jxs[22]     # ELAS: elastic angular tables
        j_ionization = ace.jxs[23]      # EION: electroionization table info
        j_brem = ace.jxs[24]            # BREMI: bremsstrahlung table info
        j_brem_tab = ace.jxs[25]        # BREME: bremsstrahlung spectrum tables
        
        data.shells = [_SUBSHELLS[int(i)] for i in ace.xss[j_shell : j_shell + n_subshells]]
        data.energy_grid = ace.xss[j_energy : j_energy + n_energy]*EV_PER_MEV
        
        j_xs = j_energy + n_energy

        # Read cross sections from the ESZE block. The layout is, in order:
        # energy grid, total, elastic, bremsstrahlung, excitation, total
        # electroionization, then one block per subshell. The total and the
        # total electroionization are deliberately skipped: the total is
        # recomputed by the transport code from the partials, and the
        # subshell cross sections are read individually below.
        #
        # Note that the elastic cross section here is the LARGE-ANGLE elastic
        # cross section, which is the quantity consistent with the ELAS
        # angular tables used for single-event transport. The transport-
        # corrected and total elastic cross sections added at JXS(27) in
        # EPRDATA14 must NOT be substituted here -- pairing either of those
        # with these angular tables would double count the small-angle
        # treatment.
        data.elastic_xs = ace.xss[j_xs + n_energy : j_xs + 2 * n_energy]
        data.bremsstrahlung_xs = ace.xss[j_xs + 2 * n_energy : j_xs + 3 * n_energy]
        data.excitation_xs = ace.xss[j_xs + 3 * n_energy : j_xs + 4 * n_energy]

        # Average excitation energy loss, from the EXCIT block at JXS(20).
        # This is NOT at JXS(5), which locates the photon heating numbers.
        data.excitation_energy_loss = Tabulated1D(
            ace.xss[j_excitation : j_excitation + n_xl],
            ace.xss[j_excitation + n_xl : j_excitation + 2 * n_xl])

        j_subshell_xs = j_xs + 5 * n_energy
        for s, shell in enumerate(data.shells):
            start_idx = j_subshell_xs + s * n_energy
            data.ionization_xs[shell] = ace.xss[start_idx : start_idx + n_energy]
            
        ni = ace.xss[j_ionization : j_ionization + n_subshells].astype(int)
        locinfo = ace.xss[j_ionization + n_subshells: j_ionization + 2*n_subshells].astype(int)
        loctab = ace.xss[j_ionization + 2*n_subshells: j_ionization + 3*n_subshells].astype(int)
        for s, shell in enumerate(data.shells):
            data.ionization_dist[shell] = UncorrelatedAngleEnergy()
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
            data.ionization_dist[shell].energy = ContinuousTabular([len(energy)],[2], energy, energy_out)
            
        
        data.bremsstrahlung_dist = UncorrelatedAngleEnergy()
        nb = ace.nxs[11]
        energy = ace.xss[j_brem:j_brem+nb]*EV_PER_MEV
        lb = ace.xss[j_brem+nb:j_brem+2*nb].astype(int)
        offsets = ace.xss[j_brem+2*nb:j_brem+3*nb].astype(int)
        energy_out = []
        for i in range(nb):
            start = j_brem_tab + offsets[i]
            e = ace.xss[start:start + lb[i]]*EV_PER_MEV
            c = ace.xss[start + lb[i]:start + 2*lb[i]]
            energy_out.append(_tabular_from_cdf(
                e, c, f'bremsstrahlung table {i}'))
        data.bremsstrahlung_dist.energy = ContinuousTabular([len(energy)],[2], energy, energy_out)
            
        na = ace.nxs[10]
        energy = ace.xss[j_elastic : j_elastic + na]*EV_PER_MEV
        le = ace.xss[j_elastic + na : j_elastic + 2 * na].astype(int)
        offsets = ace.xss[j_elastic + 2 * na : j_elastic + 3 * na].astype(int)
        mu = []
        for i in range(na):
            start = j_elastic_tab + offsets[i]
            cos = ace.xss[start:start + le[i]]
            c = ace.xss[start + le[i]:start + 2*le[i]]
            mu.append(_tabular_from_cdf(cos, c, f'elastic angular table {i}'))
        
        data.elastic_dist = AngleDistribution(energy, mu)

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
            self.bremsstrahlung_dist.to_hdf5(bremsstrahlung_group.create_group("distribution"))
