from collections.abc import Mapping
from io import StringIO
from numbers import Integral, Real

import numpy as np
import h5py

from . import HDF5_VERSION
from .data import ATOMIC_SYMBOL, gnds_name
from .endf import Evaluation, get_head_record, get_tab1_record
from .function import Tabulated1D, Sum
from .product import Product
from .reaction import Reaction
import openmc.checkvalue as cv
from openmc.mixin import EqualityMixin


class IncidentDeuteron(EqualityMixin):
    """Continuous-energy deuteron interaction data.

    This class stores data derived from an ENDF-6 format deuteron interaction
    sublibrary. Instances of this class are not normally instantiated by the
    user but rather created using the factory method
    :meth:`IncidentDeuteron.from_endf`.

    Parameters
    ----------
    name : str
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
    name : str
        Name of the target nuclide using the GNDS naming convention
    reactions : dict
        Contains the cross sections, secondary angle and energy distributions,
        and other associated data for each reaction. The keys are the MT values
        and the values are Reaction objects.

    """

    def __init__(self, name, atomic_number, mass_number, metastable,
                 atomic_weight_ratio):
        self.name = name
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
        return f"<IncidentDeuteron: {self.name}>"

    def __iter__(self):
        return iter(self.reactions.values())

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        cv.check_type('name', name, str)
        self._name = name

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
        """Export incident deuteron data to an HDF5 file.

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
        # and Q-value array
        xs_matrix = np.zeros((n_energy, n_reactions))
        q_values = np.zeros(n_reactions)

        for j, mt in enumerate(mt_list):
            rx = self.reactions[mt]
            q_values[j] = rx.q_value
            if '0K' in rx.xs and rx.xs['0K'] is not None:
                xs_func = rx.xs['0K']
                # Evaluate XS at each energy point
                for i, E in enumerate(energy):
                    val = xs_func(E)
                    xs_matrix[i, j] = val if val is not None else 0.0

        # Write to HDF5
        with h5py.File(str(path), mode, libver=libver) as f:
            f.attrs['filetype'] = np.bytes_('data_deuteron')
            f.attrs['version'] = np.array(HDF5_VERSION)

            g = f.create_group(self.name)
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

            # Write cross section matrix
            g.create_dataset('xs', data=xs_matrix)

    @classmethod
    def from_hdf5(cls, group_or_filename):
        """Generate continuous-energy deuteron interaction data from HDF5 group

        Parameters
        ----------
        group_or_filename : h5py.Group or str
            HDF5 group containing interaction data. If given as a string, it is
            assumed to be the filename for the HDF5 file, and the first group is
            used to read from.

        Returns
        -------
        openmc.data.IncidentDeuteron
            Continuous-energy deuteron interaction data

        """
        if isinstance(group_or_filename, h5py.Group):
            group = group_or_filename
            need_to_close = False
        else:
            h5file = h5py.File(str(group_or_filename), 'r')
            group = list(h5file.values())[0]
            need_to_close = True

        name = group.name[1:]
        atomic_number = group.attrs['Z']
        mass_number = group.attrs['A']
        metastable = group.attrs['metastable']
        atomic_weight_ratio = group.attrs['atomic_weight_ratio']

        data = cls(name, atomic_number, mass_number, metastable,
                   atomic_weight_ratio)

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
        """Generate incident deuteron continuous-energy data from an ENDF evaluation

        Parameters
        ----------
        ev_or_filename : openmc.data.endf.Evaluation or str
            ENDF evaluation to read from. If given as a string, it is assumed to
            be the filename for the ENDF file.

        Returns
        -------
        openmc.data.IncidentDeuteron
            Incident deuteron continuous-energy data

        """
        if isinstance(ev_or_filename, Evaluation):
            ev = ev_or_filename
        else:
            ev = Evaluation(ev_or_filename)

        atomic_number = ev.target['atomic_number']
        mass_number = ev.target['mass_number']
        metastable = ev.target['isomeric_state']
        atomic_weight_ratio = ev.target['mass']

        # Determine name
        name = gnds_name(atomic_number, mass_number, metastable)

        # Instantiate incident deuteron data
        data = cls(name, atomic_number, mass_number, metastable,
                   atomic_weight_ratio)

        # Read each reaction
        for mf, mt, nc, mod in ev.reaction_list:
            if mf == 3:
                data.reactions[mt] = Reaction.from_endf(ev, mt)

        data._evaluation = ev
        return data
