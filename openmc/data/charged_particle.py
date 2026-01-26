from collections.abc import Mapping
from io import StringIO
from numbers import Integral, Real

import numpy as np
import h5py

from . import HDF5_VERSION
from .ace import Table, get_table, get_metadata, TableType
from .data import ATOMIC_SYMBOL, gnds_name, EV_PER_MEV, K_BOLTZMANN
from .endf import Evaluation, get_head_record, get_tab1_record
from .function import Tabulated1D, Sum
from .product import Product
from .reaction import Reaction
import openmc.checkvalue as cv
from openmc.mixin import EqualityMixin


# Mapping of projectile names to (Z, A) and file prefixes
CHARGED_PARTICLES = {
    'proton': (1, 1, 'p'),
    'deuteron': (1, 2, 'd'),
    'triton': (1, 3, 't'),
    'helion': (2, 3, 'h'),  # He-3
    'alpha': (2, 4, 'a'),
}

# Reverse mapping from (Z, A) to particle name
_ZA_TO_PARTICLE = {(z, a): name for name, (z, a, _) in CHARGED_PARTICLES.items()}

# Mapping from GNDS/ENDF particle names to standard names
_PARTICLE_NAME_MAP = {
    'H1': 'proton',
    'H2': 'deuteron',
    'H3': 'triton',
    'He3': 'helion',
    'He4': 'alpha',
    'n': 'neutron',
    'neutron': 'neutron',
    'photon': 'photon',
    'gamma': 'photon',
    'proton': 'proton',
    'deuteron': 'deuteron',
    'triton': 'triton',
    'helion': 'helion',
    'alpha': 'alpha',
}


def get_particle_type(z, a):
    """Get particle name from Z and A values."""
    return _ZA_TO_PARTICLE.get((z, a), f'Z{z}A{a}')


def standardize_particle_name(name):
    """Convert particle name to standard form (proton, deuteron, etc.)."""
    return _PARTICLE_NAME_MAP.get(name, name)


class IncidentChargedParticle(EqualityMixin):
    """Continuous-energy charged particle interaction data.

    This class stores data derived from an ENDF-6 format charged particle
    interaction sublibrary (protons, deuterons, tritons, helions, or alphas).
    Instances of this class are not normally instantiated by the user but
    rather created using the factory method
    :meth:`IncidentChargedParticle.from_endf`.

    Parameters
    ----------
    projectile : str
        Type of incident particle ('proton', 'deuteron', 'triton', 'helion', 'alpha')
    target_name : str
        Name of the target nuclide using the GNDS naming convention
    atomic_number : int
        Number of protons in the target nucleus
    mass_number : int
        Number of nucleons in the target nucleus
    metastable : int
        Metastable state of the target nucleus. A value of zero indicates ground
        state.
    atomic_weight_ratio : float
        Atomic mass ratio of the target nuclide.

    Attributes
    ----------
    projectile : str
        Type of incident particle
    target_name : str
        Name of the target nuclide using the GNDS naming convention
    atomic_number : int
        Number of protons in the target nucleus
    atomic_symbol : str
        Atomic symbol of the nuclide, e.g., 'Zr'
    atomic_weight_ratio : float
        Atomic weight ratio of the target nuclide.
    mass_number : int
        Number of nucleons in the target nucleus
    metastable : int
        Metastable state of the target nucleus. A value of zero indicates ground
        state.
    reactions : dict
        Contains the cross sections, secondary angle and energy distributions,
        and other associated data for each reaction. The keys are the MT values
        and the values are Reaction objects.

    """

    def __init__(self, projectile, target_name, atomic_number, mass_number,
                 metastable, atomic_weight_ratio):
        self.projectile = projectile
        self.target_name = target_name
        self.atomic_number = atomic_number
        self.mass_number = mass_number
        self.metastable = metastable
        self.atomic_weight_ratio = atomic_weight_ratio
        self.reactions = {}

    def __contains__(self, mt):
        return mt in self.reactions

    def __getitem__(self, mt):
        if mt in self.reactions:
            return self.reactions[mt]
        else:
            raise KeyError(f'No reaction with MT={mt}.')

    def __repr__(self):
        return f"<IncidentChargedParticle: {self.projectile} on {self.target_name}>"

    def __iter__(self):
        return iter(self.reactions.values())

    @property
    def name(self):
        """Full name combining projectile and target."""
        return f"{self.projectile}_{self.target_name}"

    @property
    def projectile(self):
        return self._projectile

    @projectile.setter
    def projectile(self, projectile):
        cv.check_type('projectile', projectile, str)
        self._projectile = projectile

    @property
    def target_name(self):
        return self._target_name

    @target_name.setter
    def target_name(self, target_name):
        cv.check_type('target_name', target_name, str)
        self._target_name = target_name

    @property
    def atomic_number(self):
        return self._atomic_number

    @atomic_number.setter
    def atomic_number(self, atomic_number):
        cv.check_type('atomic number', atomic_number, Integral)
        cv.check_greater_than('atomic number', atomic_number, 0, True)
        self._atomic_number = atomic_number

    @property
    def mass_number(self):
        return self._mass_number

    @mass_number.setter
    def mass_number(self, mass_number):
        cv.check_type('mass number', mass_number, Integral)
        cv.check_greater_than('mass number', mass_number, 0, True)
        self._mass_number = mass_number

    @property
    def metastable(self):
        return self._metastable

    @metastable.setter
    def metastable(self, metastable):
        cv.check_type('metastable', metastable, Integral)
        cv.check_greater_than('metastable', metastable, 0, True)
        self._metastable = metastable

    @property
    def atomic_weight_ratio(self):
        return self._atomic_weight_ratio

    @atomic_weight_ratio.setter
    def atomic_weight_ratio(self, atomic_weight_ratio):
        cv.check_type('atomic weight ratio', atomic_weight_ratio, Real)
        cv.check_greater_than('atomic weight ratio', atomic_weight_ratio, 0.0)
        self._atomic_weight_ratio = atomic_weight_ratio

    @property
    def reactions(self):
        return self._reactions

    @reactions.setter
    def reactions(self, reactions):
        cv.check_type('reactions', reactions, Mapping)
        self._reactions = reactions

    @property
    def atomic_symbol(self):
        return ATOMIC_SYMBOL[self.atomic_number]

    def export_to_hdf5(self, path, mode='a', libver='earliest'):
        """Export incident charged particle data to an HDF5 file.

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
        # Collect all unique energies to create union grid
        all_energies = set()
        for rx in self.reactions.values():
            if '0K' in rx.xs and rx.xs['0K'] is not None:
                all_energies.update(rx.xs['0K'].x)
        energy = np.array(sorted(all_energies))

        # Get sorted MT numbers
        mt_list = np.array(sorted(self.reactions.keys()), dtype=np.int32)
        n_energy = len(energy)
        n_reactions = len(mt_list)

        # Build cross section matrix (n_energy x n_reactions)
        # and Q-value array and ejectile types
        xs_matrix = np.zeros((n_energy, n_reactions))
        q_values = np.zeros(n_reactions)
        ejectiles = []

        for j, mt in enumerate(mt_list):
            rx = self.reactions[mt]
            q_values[j] = rx.q_value
            if '0K' in rx.xs and rx.xs['0K'] is not None:
                xs_func = rx.xs['0K']
                # Evaluate XS at each energy point
                for i, E in enumerate(energy):
                    val = xs_func(E)
                    xs_matrix[i, j] = val if val is not None else 0.0

            # Get ejectile particle type from products
            # For elastic (MT=2), ejectile is same as projectile
            if mt == 2:
                ejectiles.append(self.projectile)
            elif mt == 5:
                # MT=5 is a lumped reaction with multiple possible products.
                # Don't assign a specific ejectile - use projectile to indicate
                # no secondary particle should be created (similar to elastic).
                ejectiles.append(self.projectile)
            elif rx.products:
                # First product is typically the ejectile
                ejectiles.append(standardize_particle_name(rx.products[0].particle))
            else:
                # Default to projectile if no products defined
                ejectiles.append(self.projectile)

        # Write to HDF5
        with h5py.File(str(path), mode, libver=libver) as f:
            f.attrs['filetype'] = np.bytes_('data_charged_particle')
            f.attrs['version'] = np.array(HDF5_VERSION)

            g = f.create_group(self.name)
            g.attrs['projectile'] = np.bytes_(self.projectile)
            g.attrs['target'] = np.bytes_(self.target_name)
            g.attrs['Z'] = self.atomic_number
            g.attrs['A'] = self.mass_number
            g.attrs['metastable'] = self.metastable
            g.attrs['atomic_weight_ratio'] = self.atomic_weight_ratio

            # Write energy grid
            g.create_dataset('energy', data=energy)

            # Write MT numbers
            g.create_dataset('MT', data=mt_list)

            # Write Q-values
            g.create_dataset('Q_value', data=q_values)

            # Write ejectile particle types as integers
            # Mapping: 0=neutron, 1=photon, 2=electron, 3=positron,
            #          4=proton, 5=deuteron, 6=triton, 7=helion, 8=alpha
            particle_type_map = {
                'neutron': 0, 'photon': 1, 'electron': 2, 'positron': 3,
                'proton': 4, 'deuteron': 5, 'triton': 6, 'helion': 7, 'alpha': 8
            }
            ejectile_indices = np.array(
                [particle_type_map.get(e, 0) for e in ejectiles], dtype=np.int32)
            g.create_dataset('ejectile', data=ejectile_indices)

            # Write cross section matrix
            g.create_dataset('xs', data=xs_matrix)

            # Write reactions group with full product data
            # This enables energy-dependent product yields and distributions
            rxs_group = g.create_group('reactions')
            for j, mt in enumerate(mt_list):
                rx = self.reactions[mt]
                rx_group = rxs_group.create_group(f'reaction_{mt:03d}')
                rx_group.attrs['mt'] = mt
                rx_group.attrs['Q_value'] = rx.q_value

                # Write products if available
                if rx.products:
                    for i, prod in enumerate(rx.products):
                        pgroup = rx_group.create_group(f'product_{i}')
                        try:
                            prod.to_hdf5(pgroup)
                        except (AttributeError, TypeError, ValueError) as e:
                            # If full product export fails (e.g., incompatible distribution),
                            # delete the partial group and write minimal data
                            del rx_group[f'product_{i}']
                            pgroup = rx_group.create_group(f'product_{i}')
                            # Write minimal data: particle type and yield
                            # The C++ code handles missing distributions
                            pgroup.attrs['particle'] = np.bytes_(prod.particle)
                            pgroup.attrs['emission_mode'] = np.bytes_(prod.emission_mode)
                            prod.yield_.to_hdf5(pgroup, 'yield')
                            pgroup.attrs['n_distribution'] = 0

    @classmethod
    def from_hdf5(cls, group_or_filename):
        """Generate continuous-energy charged particle data from HDF5 group

        Parameters
        ----------
        group_or_filename : h5py.Group or str
            HDF5 group containing interaction data. If given as a string, it is
            assumed to be the filename for the HDF5 file, and the first group is
            used to read from.

        Returns
        -------
        openmc.data.IncidentChargedParticle
            Continuous-energy charged particle interaction data

        """
        if isinstance(group_or_filename, h5py.Group):
            group = group_or_filename
            need_to_close = False
        else:
            h5file = h5py.File(str(group_or_filename), 'r')
            group = list(h5file.values())[0]
            need_to_close = True

        projectile = group.attrs['projectile'].decode()
        target_name = group.attrs['target'].decode()
        atomic_number = group.attrs['Z']
        mass_number = group.attrs['A']
        metastable = group.attrs['metastable']
        atomic_weight_ratio = group.attrs['atomic_weight_ratio']

        data = cls(projectile, target_name, atomic_number, mass_number,
                   metastable, atomic_weight_ratio)

        # Read energy grid, MT numbers, Q-values, and XS matrix
        energy = group['energy'][()]
        mt_list = group['MT'][()]
        q_values = group['Q_value'][()]
        xs_matrix = group['xs'][()]

        # Reconstruct reactions
        for j, mt in enumerate(mt_list):
            rx = Reaction(int(mt))
            rx.q_value = q_values[j]
            rx.xs['0K'] = Tabulated1D(energy, xs_matrix[:, j])
            data.reactions[int(mt)] = rx

        if need_to_close:
            h5file.close()

        return data

    @classmethod
    def from_endf(cls, ev_or_filename):
        """Generate incident charged particle data from an ENDF evaluation

        Parameters
        ----------
        ev_or_filename : openmc.data.endf.Evaluation or str
            ENDF evaluation to read from. If given as a string, it is assumed to
            be the filename for the ENDF file.

        Returns
        -------
        openmc.data.IncidentChargedParticle
            Incident charged particle continuous-energy data

        """
        if isinstance(ev_or_filename, Evaluation):
            ev = ev_or_filename
        else:
            ev = Evaluation(ev_or_filename)

        # Determine projectile type from sublibrary or projectile mass
        sublibrary = ev.info.get('sublibrary', '')
        projectile_mass = ev.projectile.get('mass', 1.0)

        # Determine projectile from sublibrary name or mass
        if 'proton' in sublibrary.lower():
            projectile = 'proton'
        elif 'deuteron' in sublibrary.lower():
            projectile = 'deuteron'
        elif 'triton' in sublibrary.lower():
            projectile = 'triton'
        elif 'helion' in sublibrary.lower() or '3he' in sublibrary.lower():
            projectile = 'helion'
        elif 'alpha' in sublibrary.lower():
            projectile = 'alpha'
        else:
            # Guess from projectile mass
            if abs(projectile_mass - 1.0) < 0.1:
                projectile = 'proton'
            elif abs(projectile_mass - 2.0) < 0.1:
                projectile = 'deuteron'
            elif abs(projectile_mass - 3.0) < 0.1:
                projectile = 'triton'
            elif abs(projectile_mass - 4.0) < 0.1:
                projectile = 'alpha'
            else:
                projectile = 'unknown'

        atomic_number = ev.target['atomic_number']
        mass_number = ev.target['mass_number']
        metastable = ev.target['isomeric_state']
        atomic_weight_ratio = ev.target['mass']

        # Determine target name
        target_name = gnds_name(atomic_number, mass_number, metastable)

        # Instantiate incident charged particle data
        data = cls(projectile, target_name, atomic_number, mass_number,
                   metastable, atomic_weight_ratio)

        # Read each reaction
        for mf, mt, nc, mod in ev.reaction_list:
            if mf == 3:
                data.reactions[mt] = Reaction.from_endf(ev, mt)

        data._evaluation = ev
        return data

    @classmethod
    def from_ace(cls, ace_or_filename, metastable_scheme='nndc'):
        """Generate incident charged particle data from an ACE table

        Parameters
        ----------
        ace_or_filename : openmc.data.ace.Table or str
            ACE table to read from. If the value is a string, it is assumed to
            be the filename for the ACE file.
        metastable_scheme : {'nndc', 'mcnp'}
            Determine how ZAID identifiers are to be interpreted in the case of
            a metastable nuclide. Because the normal ZAID (=1000*Z + A) does not
            encode metastable information, different conventions are used among
            different libraries. In MCNP libraries, the convention is to add 400
            for a metastable nuclide except for Am242m, for which 95242 is
            metastable and 95642 (or 1095242 in newer libraries) is the ground
            state. For NNDC libraries, ZAID is given as 1000*Z + A + 100*m.

        Returns
        -------
        openmc.data.IncidentChargedParticle
            Incident charged particle continuous-energy data

        """
        # First obtain the data for the first provided ACE table/file
        if isinstance(ace_or_filename, Table):
            ace = ace_or_filename
        else:
            ace = get_table(ace_or_filename)

        # Parse ZAID and suffix to determine projectile type
        zaid, xs = ace.name.split('.')
        suffix = xs[-1]

        # Map ACE suffix to projectile type
        suffix_to_projectile = {
            'h': 'proton',
            'o': 'deuteron',
            'r': 'triton',
            's': 'helion',
            'a': 'alpha',
        }

        if suffix not in suffix_to_projectile:
            raise TypeError(
                f"{ace} is not a charged particle ACE table. "
                f"Expected suffix in {list(suffix_to_projectile.keys())}, got '{suffix}'."
            )

        projectile = suffix_to_projectile[suffix]

        # Get target metadata from ZAID
        name, element, Z, mass_number, metastable = \
            get_metadata(int(zaid), metastable_scheme)

        # Get temperature string for indexing
        temperature_K = ace.temperature * EV_PER_MEV / K_BOLTZMANN
        strT = str(int(round(temperature_K))) + "K"

        # Create the IncidentChargedParticle instance
        data = cls(projectile, name, Z, mass_number, metastable,
                   ace.atomic_weight_ratio)

        # Read energy grid
        n_energy = ace.nxs[3]
        i = ace.jxs[1]
        energy = ace.xss[i : i + n_energy] * EV_PER_MEV

        # Read total cross section (at offset n_energy from start of ESZ block)
        total_xs = ace.xss[i + n_energy : i + 2*n_energy]

        # Read elastic cross section (at offset 3*n_energy from start of ESZ block)
        elastic_xs = ace.xss[i + 3*n_energy : i + 4*n_energy]

        # Create total reaction (MT=1) - redundant
        total = Reaction(1)
        total.xs[strT] = Tabulated1D(energy, total_xs)
        total.redundant = True
        data.reactions[1] = total

        # Create elastic reaction (MT=2)
        elastic = Reaction(2)
        # Fix negative cross sections if any
        if np.any(elastic_xs < 0.0):
            elastic_xs = elastic_xs.copy()
            elastic_xs[elastic_xs < 0.0] = 0.0
        elastic.xs[strT] = Tabulated1D(energy, elastic_xs)
        data.reactions[2] = elastic

        # Read non-elastic reactions
        n_reaction = ace.nxs[4]  # Number of reactions excluding elastic
        for i_reaction in range(n_reaction):
            # Get MT number
            mt = int(ace.xss[ace.jxs[3] + i_reaction])

            rx = Reaction(mt)

            # Get Q-value
            rx.q_value = ace.xss[ace.jxs[4] + i_reaction] * EV_PER_MEV

            # Get locator for cross-section data
            loc = int(ace.xss[ace.jxs[6] + i_reaction])

            # Determine starting index on energy grid
            threshold_idx = int(ace.xss[ace.jxs[7] + loc - 1]) - 1

            # Determine number of energies in reaction
            n_energy_rx = int(ace.xss[ace.jxs[7] + loc])

            # Get energy values for this reaction
            energy_rx = energy[threshold_idx:threshold_idx + n_energy_rx]

            # Read reaction cross section
            xs = ace.xss[ace.jxs[7] + loc + 1:ace.jxs[7] + loc + 1 + n_energy_rx]

            # Fix negative cross sections if any
            if np.any(xs < 0.0):
                xs = xs.copy()
                xs[xs < 0.0] = 0.0

            rx.xs[strT] = Tabulated1D(energy_rx, xs)

            # Set products to indicate the ejectile particle type
            # This is important for OpenMC to know what secondary particles to create
            # MT 50-91: (x,n) reactions - neutron emission to discrete levels
            # MT 600-649: (x,p) reactions - proton emission
            # MT 650-699: (x,d) reactions - deuteron emission
            # MT 700-749: (x,t) reactions - triton emission
            # MT 750-799: (x,3He) reactions - He-3 emission
            # MT 800-849: (x,a) reactions - alpha emission
            if 50 <= mt <= 91:
                # (x,n) reactions - neutron is the ejectile
                rx.products = [Product('neutron')]
            elif 600 <= mt <= 649:
                rx.products = [Product('proton')]
            elif 650 <= mt <= 699:
                rx.products = [Product('deuteron')]
            elif 700 <= mt <= 749:
                rx.products = [Product('triton')]
            elif 750 <= mt <= 799:
                rx.products = [Product('helion')]
            elif 800 <= mt <= 849:
                rx.products = [Product('alpha')]
            # For MT 4 (inelastic sum), don't set products - it's a sum reaction

            data.reactions[mt] = rx

        return data


# Convenience aliases for backwards compatibility
IncidentDeuteron = IncidentChargedParticle
