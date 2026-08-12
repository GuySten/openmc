from numbers import Integral

import numpy as np
import h5py

import openmc.checkvalue as cv
from .function import Tabulated1D
from .ace import get_metadata, Table, Library
from .data import ATOMIC_SYMBOL


class IncidentElectron:
    """Continuous-energy incident electron interaction data parsed from ACE."""

    def __init__(self, atomic_number):
        self.atomic_number = atomic_number
        self.energy_grid = None
        self.elastic_xs = None
        self.elastic_dist = None
        self.bremsstrahlung_xs = None
        self.excitation_xs = None
        self.ionization_xs = {}  # Keyed by subshell index

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

        # Parse NXS/JXS array layout
        n_energy = ace.nxs[1]
        n_subshells = ace.nxs[5]

        j_energy = ace.jxs[1]
        j_elastic = ace.jxs[2]
        j_ionization = ace.jxs[3]
        j_brem = ace.jxs[4]
        j_excitation = ace.jxs[5]
        j_excitation_loss = ace.jxs[20]
        j_shell = ace.jxs[11]
        
        data.shells = [_SUBSHELLS[int(i)] for i in ace.xss[ace.jxs[11] : ace.jxs[11]+n_subshells]]

        # Extract underlying XSS master arrays
        # Subtraction accounts for Python's 0-indexed slicing vs FORTRAN 1-indexed
        data.energy_grid = ace.xss[j_energy - 1 : j_energy - 1 + n_energy]

        # Read Cross Sections
        data.elastic_xs = ace.xss[j_elastic - 1 : j_elastic - 1 + n_energy]
        data.bremsstrahlung_xs = ace.xss[j_brem - 1 : j_brem - 1 + n_energy]
        data.excitation_xs = ace.xss[j_excitation - 1 : j_excitation - 1 + n_energy]
        data.excitation_energy_loss = ace.xss[j_excitation_loss - 1 : j_excitation_loss - 1 + n_energy]

        # Read Subshell Ionization Data Blocks
        idx = j_ionization - 1
        for shell in data.shells:
            # Parse individual binding energies and localized grids if variable
            subshell_xs = ace.xss[idx : idx + n_energy]
            
            data.ionization_xs[shell] = subshell_xs
            idx += n_energy
            
        self.elastic_dist = ElasticAngularDist.from_ace(ace)

        return data

    def export_to_hdf5(self, path, mode="a", libver="earliest"):
        """Export incident photon data to an HDF5 file.

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
            dist_group = elastic_group.create_group("distribution")
            self.elastic_dist.to_hdf5(ang_group)
            
            excitation_group = group.create_group("excitation")
            excitation_group.create_dataset("xs", data=self.excitation_xs)
            excitation_group.create_dataset("energy_loss", data=self.excitation_energy_loss)
            
            ionization_group = group.create_group("ionization")
            ionization_group.attrs['designators'] = np.array(self.shells, dtype='S')
            xs = np.zeros((len(self.shells), len(self.energy_grid)))
            for i, shell in enumerate(self.shells):
                xs[i] = self.ionization_xs[shell]
            ionization_group.create_dataset("xs", data=xs)
            
            bremsstrahlung_group = group.create_group("bremsstrahlung")
            bremsstrahlung_group.create_dataset("xs", data=self.bremsstrahlung_xs)



class ElasticAngularDist:
    """Contains low-energy tabular distributions and high-energy analytical moments."""

    def __init__(self):
        self.n_legendre_coefficients = 0
        self.tabular_energies = None
        self.tabular_pdf = None  # Discrete angles
        self.legendre_coefficients = None  # Energy-dependent moments

    @classmethod
    def from_ace(cls, ace):
        """Parses elastic scattering distribution blocks from an ACE table object.

        Parameters
        ----------
        ace : openmc.data.ace.Table
            The underlying ACE table object being parsed.
        """
        j_elastic = ace.jxs[1]  # JXS(2) corresponds to elastic scattering data
        if j_elastic == 0:
            return None

        instance = cls()

        # 1. Read base structural dimensions from the XSS array
        idx = j_elastic - 1
        instance.n_legendre_coefficients = int(ace.xss[idx])
        n_tabular_energies = int(ace.xss[idx + 1])

        # Advance slice index past the internal block structural headers
        idx += 2

        # 2. Extract Low-Energy Tabular Angular Data (Screened Rutherford Distribution)
        if n_tabular_energies > 0:
            # Extract the specialized energy grid for these tabular data points
            instance.tabular_energies = ace.xss[idx : idx + n_tabular_energies]
            idx += n_tabular_energies

            # Parse angular bins for each tabulated energy step
            # EPRDATA14 files use 32 equiprobable bin boundaries
            n_bins = 32
            total_tab_elements = n_tabular_energies * n_bins
            instance.tabular_pdf = ace.xss[idx : idx + total_tab_elements].reshape(
                n_tabular_energies, n_bins
            )
            idx += total_tab_elements

        # 3. Extract High-Energy Legendre Moments (Goudsmit-Saunderson Data)
        if instance.n_legendre_coefficients > 0:
            # Total energy grid points (NXS(1)) minus the low energy points
            n_energy_grid = ace.nxs[0]
            n_high_energies = n_energy_grid - n_tabular_energies
            total_leg_elements = n_high_energies * instance.n_legendre_coefficients

            instance.legendre_coefficients = ace.xss[
                idx : idx + total_leg_elements
            ].reshape(n_high_energies, instance.n_legendre_coefficients)

        return instance

    def to_hdf5(self, group):
        """Write the angular distribution datasets into an OpenMC HDF5 group."""
        group.attrs["n_legendre_coefficients"] = self.n_legendre_coefficients

        if self.tabular_energies is not None:
            tab_group = group.create_group("tabular")
            tab_group.create_dataset("energies", data=self.tabular_energies)
            tab_group.create_dataset("pdf_bins", data=self.tabular_pdf)

        if self.legendre_coefficients is not None:
            leg_group = group.create_group("legendre")
            leg_group.create_dataset("coefficients", data=self.legendre_coefficients)
