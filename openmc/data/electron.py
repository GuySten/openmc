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
        self.elastic_transport_xs = None
        self.elastic_total_xs = None
        self.bremsstrahlung_xs = None
        self.bremsstrahlung_dist = None
        self.bremsstrahlung_mean_energy = None
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
        # later. EPRDATA12 is rejected because it has no JXS(27) block, and
        # without the transport-corrected and total elastic cross sections
        # stored there the elastic scattering cannot be sampled correctly.
        format_flag = ace.nxs[6]
        if format_flag != 3:
            raise ValueError(
                f'Unsupported electron-photon-relaxation format flag '
                f'NXS(6)={format_flag} in table {ace.name}. EPRDATA14 or later '
                '(NXS(6)=3) is required for the elastic cross sections at '
                'JXS(27).')

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
        j_brem_mean = ace.jxs[26]       # BREML: average emitted photon energy
        
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

        # EPRDATA14 adds a block at JXS(27) holding, on this same dense energy
        # grid, the transport-corrected elastic cross section followed by the
        # total elastic cross section. The transport cross section is the first
        # moment of the total, so its ratio to the total is the mean deflection
        # 1-<mu>; where an angular table exists the two agree to better than 4%.
        #
        # Both are worth carrying because the angular tables are tabulated far
        # too sparsely to interpolate: for aluminium there is no table between
        # 256 keV and 10 MeV, and 1-<mu> falls by a factor of 35 across that
        # gap. This gives the correct value on the 373-point grid instead of
        # requiring it to be guessed between two distant tables. The total is
        # needed alongside it to separate the forward peak, which the angular
        # tables do not cover, from the large-angle part that they do.
        j_transport = ace.jxs[27]
        if j_transport <= 0:
            raise ValueError(
                f'Table {ace.name} declares the EPRDATA14 format but has no '
                'JXS(27) elastic cross section block.')
        data.elastic_transport_xs = ace.xss[
            j_transport : j_transport + n_energy]
        data.elastic_total_xs = ace.xss[
            j_transport + n_energy : j_transport + 2 * n_energy]
        data.bremsstrahlung_xs = ace.xss[j_xs + 2 * n_energy : j_xs + 3 * n_energy]
        data.excitation_xs = ace.xss[j_xs + 3 * n_energy : j_xs + 4 * n_energy]

        # Average excitation energy loss, from the EXCIT block at JXS(20).
        # This is NOT at JXS(5), which locates the photon heating numbers.
        # Both the abscissa and the tabulated loss are stored in MeV.
        data.excitation_energy_loss = Tabulated1D(
            ace.xss[j_excitation : j_excitation + n_xl]*EV_PER_MEV,
            ace.xss[j_excitation + n_xl : j_excitation + 2 * n_xl]*EV_PER_MEV)

        # Average energy of the emitted bremsstrahlung photon, from the BREML
        # block at JXS(26): NXS(12) energies followed by NXS(12) mean energies,
        # both in MeV. This is the first moment of the same spectra stored in
        # BREME, but on a grid dense enough to interpolate -- BREME has only
        # nine incident energies between 10 eV and 100 GeV, with nothing at all
        # between 12.25 MeV and 100 GeV, and the sampled mean drifts several
        # percent off this curve in between. The transport code uses it to
        # rescale the sampled photon energy, exactly as it uses the JXS(27)
        # transport cross section to rescale the sampled elastic deflection.
        n_brem_mean = ace.nxs[12]
        if j_brem_mean > 0 and n_brem_mean > 0:
            brem_mean_e = ace.xss[j_brem_mean : j_brem_mean + n_brem_mean]
            brem_mean_k = ace.xss[
                j_brem_mean + n_brem_mean : j_brem_mean + 2 * n_brem_mean]
            # Guard against a mis-sized block rather than silently anchoring
            # the sampling to whatever happened to sit at that offset.
            if (len(brem_mean_e) == n_brem_mean
                    and np.all(np.diff(brem_mean_e) > 0.0)
                    and np.all(brem_mean_k > 0.0)
                    and np.all(brem_mean_k < brem_mean_e)):
                # ACE carries no interpolation law for this block. The mean
                # photon energy is a positive, smooth, near-power-law function
                # on a geometric grid of about seven points per decade, so
                # log-log is the faithful choice; lin-lin across a decade-wide
                # step would bias it high wherever the curve is convex.
                data.bremsstrahlung_mean_energy = Tabulated1D(
                    brem_mean_e*EV_PER_MEV, brem_mean_k*EV_PER_MEV,
                    breakpoints=[n_brem_mean], interpolation=[5])

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
            data.ionization_dist[shell].energy = ContinuousTabular(
                [len(energy)], [5], energy, energy_out)
            
        
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
        # Log-log between incident energies; see the note on the
        # electroionization tables above.
        data.bremsstrahlung_dist.energy = ContinuousTabular(
            [len(energy)], [5], energy, energy_out)
            
        na = ace.nxs[10]
        energy = ace.xss[j_elastic : j_elastic + na]*EV_PER_MEV
        le = ace.xss[j_elastic + na : j_elastic + 2 * na].astype(int)
        offsets = ace.xss[j_elastic + 2 * na : j_elastic + 3 * na].astype(int)
        mu = []
        for i in range(na):
            start = j_elastic_tab + offsets[i]
            cos = ace.xss[start:start + le[i]]
            c = ace.xss[start + le[i]:start + 2*le[i]]
            # The evaluation stops the tabulated distribution at 1 - 1e-6 and
            # leaves the forward peak beyond it to an analytic screened
            # Rutherford form. The transport code assumes that cutoff when it
            # subtracts the peak's contribution from the transport cross
            # section, so a table that ended anywhere else would be silently
            # mistreated.
            if abs(cos[-1] - (1.0 - 1.0e-6)) > 1.0e-9:
                raise ValueError(
                    f'Elastic angular table {i} of {ace.name} ends at '
                    f'mu={cos[-1]!r}, not at the 1-1e-6 cutoff the elastic '
                    'peak treatment assumes.')
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
            elastic_group.create_dataset(
                "xs_transport", data=self.elastic_transport_xs)
            elastic_group.create_dataset(
                "xs_total", data=self.elastic_total_xs)
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
            if self.bremsstrahlung_mean_energy is not None:
                self.bremsstrahlung_mean_energy.to_hdf5(
                    bremsstrahlung_group, "mean_energy")
