import numpy as np
import h5py
from .data import Tabulated1D
from .ace import Table

class IncidentElectron:
        """Continuous-energy incident electron interaction data parsed from ACE."""

    def __init__(self, atomic_number):
        self.atomic_number = atomic_number
        self.energy_grid = None
        self.elastic_xs = None
        self.bremsstrahlung_xs = None
        self.excitation_xs = None
        self.ionization_xs = {}  # Keyed by subshell index

    @classmethod
    def from_ace(cls, ace_table_or_filename):
        """Parse an eprdata14 electron-containing table."""
        if isinstance(ace_table_or_filename, Table):
            ace = ace_table_or_filename
        else:
            from .ace import Library
            lib = Library(ace_table_or_filename)
            ace = lib.tables[0]

        # Initialize instance using ZA or Atomic Number
        atomic_number = ace.zaid // 1000
        data = cls(atomic_number)

        # Parse NXS/JXS array layout
        n_energy = ace.nxs[1]
        n_subshells = ace.nxs[5]

        j_energy = ace.jxs[1]
        j_elastic = ace.jxs[2]
        j_ionization = ace.jxs[3]
        j_brem = ace.jxs[4]
        j_excitation = ace.jxs[5]

        # Extract underlying XSS master arrays
        # Subtraction accounts for Python's 0-indexed slicing vs FORTRAN 1-indexed
        data.energy_grid = ace.xss[j_energy - 1 : j_energy - 1 + n_energy]

        # Read Cross Sections (Convert log scales if necessary, check eprdata14 spec)
        data.elastic_xs = ace.xss[j_elastic - 1 : j_elastic - 1 + n_energy]
        data.bremsstrahlung_xs = ace.xss[j_brem - 1 : j_brem - 1 + n_energy]
        data.excitation_xs = ace.xss[j_excitation - 1 : j_excitation - 1 + n_energy]

        # Read Subshell Ionization Data Blocks
        idx = j_ionization - 1
        for s in range(n_subshells):
            # Parse individual binding energies and localized grids if variable
            subshell_xs = ace.xss[idx : idx + n_energy]
            data.ionization_xs[f"subshell_{s+1}"] = subshell_xs
            idx += n_energy

        return data


    def export_to_hdf5(self, path_or_group, mode='w'):
        """Write data instance contents out to an HDF5 group layer."""
        if isinstance(path_or_group, h5py.Group):
            group = path_or_group
        else:
            f = h5py.File(path_or_group, mode)
            group = f.create_group(f"z{self.atomic_number}")

        # Metadata
        group.attrs['atomic_number'] = self.atomic_number

        # Master energy grid
        group.create_dataset('energy', data=self.energy_grid)

        # Total individual cross section datasets
        xs_group = group.create_group('cross_sections')
        xs_group.create_dataset('elastic', data=self.elastic_xs)
        xs_group.create_dataset('bremsstrahlung', data=self.bremsstrahlung_xs)
        xs_group.create_dataset('excitation', data=self.excitation_xs)

        # Subshell nested breakdown
        sub_group = xs_group.create_group('ionization')
        for subshell_name, xs_array in self.ionization_xs.items():
            sub_group.create_dataset(subshell_name, data=xs_array)
        


class ElectronAngularDistribution:
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
            instance.tabular_pdf = ace.xss[idx : idx + total_tab_elements].reshape(n_tabular_energies, n_bins)
            idx += total_tab_elements

        # 3. Extract High-Energy Legendre Moments (Goudsmit-Saunderson Data)
        if instance.n_legendre_coefficients > 0:
            # Total energy grid points (NXS(1)) minus the low energy points
            n_energy_grid = ace.nxs[0]
            n_high_energies = n_energy_grid - n_tabular_energies
            total_leg_elements = n_high_energies * instance.n_legendre_coefficients

            instance.legendre_coefficients = ace.xss[idx : idx + total_leg_elements].reshape(n_high_energies, instance.n_legendre_coefficients)

        data.angular_distribution = ElectronAngularDistribution.from_ace(ace)
        return instance

    def to_hdf5(self, group):
        """Write the angular distribution datasets into an OpenMC HDF5 group."""
        group.attrs['n_legendre_coefficients'] = self.n_legendre_coefficients

        if self.tabular_energies is not None:
            tab_group = group.create_group('tabular')
            tab_group.create_dataset('energies', data=self.tabular_energies)
            tab_group.create_dataset('pdf_bins', data=self.tabular_pdf)

        if self.legendre_coefficients is not None:
            leg_group = group.create_group('legendre')
            leg_group.create_dataset('coefficients', data=self.legendre_coefficients)
        if self.angular_distribution:
            ang_group = group.create_group('angular_distribution')
            self.angular_distribution.to_hdf5(ang_group)

            
