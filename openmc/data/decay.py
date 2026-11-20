from collections.abc import Iterable
from functools import cached_property
from io import StringIO
from math import log
from warnings import warn

import h5py
import numpy as np
from uncertainties import ufloat, UFloat

import openmc
import openmc.checkvalue as cv
from openmc.exceptions import DataError
from openmc.mixin import EqualityMixin
from openmc.stats import Discrete, Tabular, Univariate, combine_distributions
from . import HDF5_VERSION
from .data import gnds_name, zam
from .function import Tabulated1D
from .function import INTERPOLATION_SCHEME
from .endf import (
    as_evaluation, get_head_record, get_list_record, get_tab1_record)


# Gives name and (change in A, change in Z) resulting from decay
_DECAY_MODES = {
    0: ('gamma', (0, 0)),
    1: ('beta-', (0, 1)),
    2: ('ec/beta+', (0, -1)),
    3: ('IT', (0, 0)),
    4: ('alpha', (-4, -2)),
    5: ('n', (-1, 0)),
    6: ('sf', None),
    7: ('p', (-1, -1)),
    8: ('e-', (0, 0)),
    9: ('xray', (0, 0)),
    10: ('unknown', None)
}

_RADIATION_TYPES = {
    0: 'gamma',
    1: 'beta-',
    2: 'ec/beta+',
    4: 'alpha',
    5: 'n',
    6: 'sf',
    7: 'p',
    8: 'e-',
    9: 'xray',
    10: 'anti-neutrino',
    11: 'neutrino'
}


def get_decay_modes(value):
    """Return sequence of decay modes given an ENDF RTYP value.

    Parameters
    ----------
    value : float
        ENDF definition of sequence of decay modes

    Returns
    -------
    list of str
        List of successive decays, e.g. ('beta-', 'neutron')

    """
    if int(value) == 10:
        # The logic below would treat 10.0 as [1, 0] rather than [10] as it
        # should, so we handle this case separately
        return ['unknown']
    else:
        return [_DECAY_MODES[int(x)][0] for x in
                str(value).strip('0').replace('.', '')]


def _uncertain_array(values):
    """Values with uncertainties as an (n, 2) array of (value, standard
    deviation); None (not given) becomes NaN"""
    return np.array([(v.nominal_value, v.std_dev) if v is not None else
                     (np.nan, np.nan) for v in values], dtype=float).reshape(-1, 2)


def _uncertain_values(array):
    """Inverse of :func:`_uncertain_array`"""
    return [None if np.isnan(v) else ufloat(v, dv) for v, dv in array]


def _hdf5_group(group_or_filename, filetypes):
    """Open an HDF5 data file of one of the given filetypes

    Returns the group (the given one, or the first group of the file), the
    open file to close (None for a given group) and the filetype.

    """
    if isinstance(group_or_filename, h5py.Group):
        group, h5file = group_or_filename, None
    else:
        h5file = h5py.File(str(group_or_filename), 'r')
        group = h5file[next(iter(h5file))]
    filetype = group.file.attrs['filetype'].decode()
    if filetype not in filetypes:
        if h5file is not None:
            h5file.close()
        raise DataError(f"'{filetype}' file is not one of {filetypes}.")
    return group, h5file, filetype


class FissionProductYields(EqualityMixin):
    """Independent and cumulative fission product yields.

    Parameters
    ----------
    ev_or_filename : str, openmc.data.endf.Evaluation, or endf.Material
        ENDF fission product yield evaluation to read from. If given as a
        string, it is assumed to be the filename for the ENDF file.

    Attributes
    ----------
    cumulative : list of dict
        Cumulative yields for each tabulated energy. Each item in the list is a
        dictionary whose keys are nuclide names and values are cumulative
        yields. The i-th dictionary corresponds to the i-th incident neutron
        energy.
    energies : Iterable of float or None
        Energies at which fission product yields are tabulated.
    interpolation : dict
        ENDF interpolation flag between the i-th and the preceding energy, for
        'independent' and 'cumulative' yields.

        .. versionadded:: 0.16.1
    independent : list of dict
        Independent yields for each tabulated energy. Each item in the list is a
        dictionary whose keys are nuclide names and values are independent
        yields. The i-th dictionary corresponds to the i-th incident neutron
        energy.
    nuclide : dict
        Properties of the fissioning nuclide.
    sublibrary : {'nfy', 'sfy'}
        Neutron-induced or spontaneous fission yields.

        .. versionadded:: 0.16.1

    Notes
    -----
    Neutron fission yields are typically not measured with a monoenergetic
    source of neutrons. As such, if the fission yields are given at, e.g.,
    0.0253 eV, one should interpret this as meaning that they are derived from a
    typical thermal reactor flux spectrum as opposed to a monoenergetic source
    at 0.0253 eV.

    """
    def __init__(self, ev_or_filename):
        # Define function that can be used to read both independent and
        # cumulative yields
        def get_yields(file_obj):
            # Determine number of energies
            n_energy = get_head_record(file_obj)[2]
            energies = np.zeros(n_energy)

            data = []
            interpolation = []
            for i in range(n_energy):
                # Determine i-th energy, interpolation flag and number of products
                items, values = get_list_record(file_obj)
                energies[i] = items[0]
                interpolation.append(items[2])
                n_products = items[5]

                # Get yields for i-th energy
                yields = {}
                for j in range(n_products):
                    Z, A = divmod(int(values[4*j]), 1000)
                    isomeric_state = int(values[4*j + 1])
                    name = gnds_name(Z, A, isomeric_state)
                    yield_j = ufloat(values[4*j + 2], values[4*j + 3])
                    yields[name] = yield_j

                data.append(yields)

            return energies, data, interpolation

        ev = as_evaluation(ev_or_filename)
        sublibraries = {'Neutron-induced fission product yields': 'nfy',
                        'Spontaneous fission product yields': 'sfy'}
        self.sublibrary = sublibraries.get(ev.info['sublibrary'], 'nfy')
        self.interpolation = {}

        # Assign basic nuclide properties
        self.nuclide = {
            'name': ev.gnds_name,
            'atomic_number': ev.target['atomic_number'],
            'mass_number': ev.target['mass_number'],
            'isomeric_state': ev.target['isomeric_state']
        }

        # Read independent yields (MF=8, MT=454)
        if (8, 454) in ev.section:
            file_obj = StringIO(ev.section[8, 454])
            self.energies, self.independent, self.interpolation['independent'] = \
                get_yields(file_obj)

        # Read cumulative yields (MF=8, MT=459)
        if (8, 459) in ev.section:
            file_obj = StringIO(ev.section[8, 459])
            energies, self.cumulative, self.interpolation['cumulative'] = \
                get_yields(file_obj)
            assert np.all(energies == self.energies)

    @classmethod
    def from_endf(cls, ev_or_filename):
        """Generate fission product yield data from an ENDF evaluation

        Parameters
        ----------
        ev_or_filename : str, openmc.data.endf.Evaluation, or endf.Material
            ENDF fission product yield evaluation to read from. If given as a
            string, it is assumed to be the filename for the ENDF file.

        Returns
        -------
        openmc.data.FissionProductYields
            Fission product yield data

        """
        return cls(ev_or_filename)

    def export_to_hdf5(self, path, mode='a', libver='earliest'):
        """Export fission product yields to an HDF5 file

        One file per fissioning nuclide, as in the ENDF sublibrary; every
        yield keeps its standard deviation.

        .. versionadded:: 0.16.1

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
            f.attrs['filetype'] = np.bytes_('data_' + self.sublibrary)
            f.attrs['version'] = np.array(HDF5_VERSION)
            g = f.create_group(self.nuclide['name'])
            for key in ('atomic_number', 'mass_number', 'isomeric_state'):
                g.attrs[key] = self.nuclide[key]
            g.create_dataset('energies', data=np.asarray(self.energies, dtype=float))
            for kind in ('independent', 'cumulative'):
                if not hasattr(self, kind):
                    continue
                k = g.create_group(kind)
                k.create_dataset('interpolation', data=self.interpolation[kind])
                for i, yields in enumerate(getattr(self, kind)):
                    e = k.create_group(f'energy_{i}')
                    e.create_dataset('products', data=np.array(
                        list(yields), dtype=h5py.string_dtype()))
                    e.create_dataset('yields', data=_uncertain_array(yields.values()))

    @classmethod
    def from_hdf5(cls, group_or_filename):
        """Read fission product yields from an HDF5 file

        .. versionadded:: 0.16.1

        Parameters
        ----------
        group_or_filename : h5py.Group or str
            HDF5 group containing the yields. If given as a string, it is
            assumed to be the filename of the HDF5 file, and its first group
            is read.

        Returns
        -------
        openmc.data.FissionProductYields
            Fission product yield data

        """
        group, h5file, filetype = _hdf5_group(group_or_filename,
                                              ('data_nfy', 'data_sfy'))
        try:
            obj = cls.__new__(cls)
            obj.sublibrary = filetype[5:]
            obj.nuclide = {key: int(group.attrs[key]) for key in
                           ('atomic_number', 'mass_number', 'isomeric_state')}
            obj.nuclide['name'] = group.name.lstrip('/')
            obj.energies = group['energies'][()]
            obj.interpolation = {}
            for kind in ('independent', 'cumulative'):
                if kind not in group:
                    continue
                k = group[kind]
                obj.interpolation[kind] = [int(i) for i in k['interpolation'][()]]
                data = []
                for i in range(len(obj.energies)):
                    e = k[f'energy_{i}']
                    names = [p.decode() for p in e['products'][()]]
                    data.append(dict(zip(names, _uncertain_values(e['yields'][()]))))
                setattr(obj, kind, data)
        finally:
            if h5file is not None:
                h5file.close()
        return obj


class DecayMode(EqualityMixin):
    """Radioactive decay mode.

    Parameters
    ----------
    parent : str
        Parent decaying nuclide
    modes : list of str
        Successive decay modes
    daughter_state : int
        Metastable state of the daughter nuclide
    energy : uncertainties.UFloat
        Total decay energy in eV available in the decay process.
    branching_ratio : uncertainties.UFloat
        Fraction of the decay of the parent nuclide which proceeds by this mode.

    Attributes
    ----------
    branching_ratio : uncertainties.UFloat
        Fraction of the decay of the parent nuclide which proceeds by this mode.
    daughter : str
        Name of daughter nuclide produced from decay
    energy : uncertainties.UFloat
        Total decay energy in eV available in the decay process.
    modes : list of str
        Successive decay modes
    parent : str
        Parent decaying nuclide

    """

    def __init__(self, parent, modes, daughter_state, energy,
                 branching_ratio):
        self._daughter_state = daughter_state
        self.parent = parent
        self.modes = modes
        self.energy = energy
        self.branching_ratio = branching_ratio

    def __repr__(self):
        return ('<DecayMode: ({}), {} -> {}, {}>'.format(
            ','.join(self.modes), self.parent, self.daughter,
            self.branching_ratio))

    @property
    def branching_ratio(self):
        return self._branching_ratio

    @branching_ratio.setter
    def branching_ratio(self, branching_ratio):
        cv.check_type('branching ratio', branching_ratio, UFloat)
        cv.check_greater_than('branching ratio',
                              branching_ratio.nominal_value, 0.0, True)
        if branching_ratio.nominal_value == 0.0:
            warn('Decay mode {} of parent {} has a zero branching ratio.'
                 .format(self.modes, self.parent))
        cv.check_greater_than('branching ratio uncertainty',
                              branching_ratio.std_dev, 0.0, True)
        self._branching_ratio = branching_ratio

    @property
    def daughter(self):
        # Determine atomic number and mass number of parent
        Z, A, _ = zam(self.parent)

        # Process changes
        for mode in self.modes:
            for name, changes in _DECAY_MODES.values():
                if name == mode:
                    if changes is not None:
                        delta_A, delta_Z = changes
                        A += delta_A
                        Z += delta_Z
                        break
                    else:
                        return None

        return gnds_name(Z, A, self._daughter_state)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        cv.check_type('parent nuclide', parent, str)
        self._parent = parent

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        cv.check_type('decay energy', energy, UFloat)
        cv.check_greater_than('decay energy', energy.nominal_value, 0.0, True)
        cv.check_greater_than('decay energy uncertainty',
                              energy.std_dev, 0.0, True)
        self._energy = energy

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self, modes):
        cv.check_type('decay modes', modes, Iterable, str)
        self._modes = modes


class Decay(EqualityMixin):
    """Radioactive decay data.

    Parameters
    ----------
    ev_or_filename : str, openmc.data.endf.Evaluation, or endf.Material
        ENDF radioactive decay data evaluation to read from. If given as a
        string, it is assumed to be the filename for the ENDF file.

    Attributes
    ----------
    average_energies : dict
        Average decay energies in eV of each type of radiation for decay heat
        applications.
    decay_constant : uncertainties.UFloat
        Decay constant in inverse seconds.
    decay_energy : uncertainties.UFloat
        Average energy in [eV] per decay for decay heat applications
    half_life : uncertainties.UFloat
        Half-life of the decay in seconds.
    modes : list
        Decay mode information for each mode of decay.
    nuclide : dict
        Dictionary describing decaying nuclide with keys 'name',
        'atomic_number', 'mass_number', 'isomeric_state', 'excited_state',
        'excitation_energy' (in [eV]), 'mass', 'stable', 'spin', and 'parity'.
    spectra : dict
        Resulting radiation spectra for each radiation type.
    sources : dict
        Radioactive decay source distributions represented as a dictionary
        mapping particle types (e.g., 'photon') to instances of
        :class:`openmc.stats.Univariate`.

        .. versionadded:: 0.13.1

    """
    def __init__(self, ev_or_filename):
        ev = as_evaluation(ev_or_filename)

        file_obj = StringIO(ev.section[8, 457])

        self.nuclide = {}
        self.modes = []
        self.spectra = {}
        self.average_energies = {}

        # Get head record
        items = get_head_record(file_obj)
        Z, A = divmod(items[0], 1000)
        metastable = items[3]
        self.nuclide['atomic_number'] = Z
        self.nuclide['mass_number'] = A
        self.nuclide['isomeric_state'] = metastable
        self.nuclide['name'] = gnds_name(Z, A, metastable)
        self.nuclide['mass'] = items[1]  # AWR
        self.nuclide['excited_state'] = items[2]  # State of the original nuclide
        self.nuclide['excitation_energy'] = ev.target['excitation_energy']  # ELIS
        self.nuclide['stable'] = (items[4] == 1)  # Nucleus stability flag

        # Determine if radioactive/stable
        if not self.nuclide['stable']:
            NSP = items[5]  # Number of radiation types

            # Half-life and decay energies
            items, values = get_list_record(file_obj)
            self.half_life = ufloat(items[0], items[1])
            NC = items[4]//2
            pairs = list(zip(values[::2], values[1::2]))
            ex = self.average_energies
            ex['light'] = ufloat(*pairs[0])
            ex['electromagnetic'] = ufloat(*pairs[1])
            ex['heavy'] = ufloat(*pairs[2])
            if NC == 17:
                ex['beta-'] = ufloat(*pairs[3])
                ex['beta+'] = ufloat(*pairs[4])
                ex['auger'] = ufloat(*pairs[5])
                ex['conversion'] = ufloat(*pairs[6])
                ex['gamma'] = ufloat(*pairs[7])
                ex['xray'] = ufloat(*pairs[8])
                ex['bremsstrahlung'] = ufloat(*pairs[9])
                ex['annihilation'] = ufloat(*pairs[10])
                ex['alpha'] = ufloat(*pairs[11])
                ex['recoil'] = ufloat(*pairs[12])
                ex['SF'] = ufloat(*pairs[13])
                ex['neutron'] = ufloat(*pairs[14])
                ex['proton'] = ufloat(*pairs[15])
                ex['neutrino'] = ufloat(*pairs[16])

            items, values = get_list_record(file_obj)
            spin = items[0]
            # ENDF-102 specifies that unknown spin should be reported as -77.777
            if spin == -77.777:
                self.nuclide['spin'] = None
            else:
                self.nuclide['spin'] = spin
            self.nuclide['parity'] = items[1]  # Parity of the nuclide

            # Decay mode information
            n_modes = items[5]  # Number of decay modes
            for i in range(n_modes):
                decay_type = get_decay_modes(values[6*i])
                isomeric_state = int(values[6*i + 1])
                energy = ufloat(*values[6*i + 2:6*i + 4])
                branching_ratio = ufloat(*values[6*i + 4:6*(i + 1)])

                mode = DecayMode(self.nuclide['name'], decay_type, isomeric_state,
                                 energy, branching_ratio)
                self.modes.append(mode)

            discrete_type = {0.0: None, 1.0: 'allowed', 2.0: 'first-forbidden',
                             3.0: 'second-forbidden', 4.0: 'third-forbidden',
                             5.0: 'fourth-forbidden', 6.0: 'fifth-forbidden'}

            # Read spectra
            for i in range(NSP):
                spectrum = {}

                items, values = get_list_record(file_obj)
                # Decay radiation type
                spectrum['type'] = _RADIATION_TYPES[items[1]]
                # Continuous spectrum flag
                spectrum['continuous_flag'] = {0: 'discrete', 1: 'continuous',
                                               2: 'both'}[items[2]]
                spectrum['discrete_normalization'] = ufloat(*values[0:2])
                spectrum['energy_average'] = ufloat(*values[2:4])
                spectrum['continuous_normalization'] = ufloat(*values[4:6])

                NER = items[5]  # Number of tabulated discrete energies

                if not spectrum['continuous_flag'] == 'continuous':
                    # Information about discrete spectrum
                    spectrum['discrete'] = []
                    for j in range(NER):
                        items, values = get_list_record(file_obj)
                        di = {}
                        di['energy'] = ufloat(*items[0:2])
                        di['from_mode'] = get_decay_modes(values[0])
                        di['type'] = discrete_type[values[1]]
                        di['intensity'] = ufloat(*values[2:4])
                        if spectrum['type'] == 'ec/beta+':
                            di['positron_intensity'] = ufloat(*values[4:6])
                        elif spectrum['type'] == 'gamma':
                            if len(values) >= 6:
                                di['internal_pair'] = ufloat(*values[4:6])
                            if len(values) >= 8:
                                di['total_internal_conversion'] = ufloat(*values[6:8])
                            if len(values) == 12:
                                di['k_shell_conversion'] = ufloat(*values[8:10])
                                di['l_shell_conversion'] = ufloat(*values[10:12])
                        spectrum['discrete'].append(di)

                if not spectrum['continuous_flag'] == 'discrete':
                    # Read continuous spectrum
                    ci = {}
                    params, ci['probability'] = get_tab1_record(file_obj)
                    ci['from_mode'] = get_decay_modes(params[0])

                    # Read covariance (Ek, Fk) table
                    LCOV = params[3]
                    if LCOV != 0:
                        items, values = get_list_record(file_obj)
                        ci['covariance_lb'] = items[3]
                        ci['covariance'] = list(zip(values[0::2], values[1::2]))

                    spectrum['continuous'] = ci

                # Add spectrum to dictionary
                self.spectra[spectrum['type']] = spectrum

        else:
            items, values = get_list_record(file_obj)
            items, values = get_list_record(file_obj)
            self.nuclide['spin'] = items[0]
            self.nuclide['parity'] = items[1]
            self.half_life = ufloat(float('inf'), float('inf'))

    @property
    def decay_constant(self):
        if self.half_life.n == 0.0:
            name = self.nuclide['name']
            raise ValueError(f"{name} is listed as unstable but has a zero half-life.")
        return log(2.)/self.half_life

    @property
    def decay_energy(self):
        energy = self.average_energies
        if energy:
            return energy['light'] + energy['electromagnetic'] + energy['heavy']
        else:
            return ufloat(0, 0)

    @classmethod
    def from_endf(cls, ev_or_filename):
        """Generate radioactive decay data from an ENDF evaluation

        Parameters
        ----------
        ev_or_filename : str, openmc.data.endf.Evaluation, or endf.Material
            ENDF radioactive decay data evaluation to read from. If given as a
            string, it is assumed to be the filename for the ENDF file.

        Returns
        -------
        openmc.data.Decay
            Radioactive decay data

        """
        return cls(ev_or_filename)

    # optional (value, uncertainty) columns of discrete lines
    _LINE_COLUMNS = ('positron_intensity', 'internal_pair',
                     'total_internal_conversion', 'k_shell_conversion',
                     'l_shell_conversion')

    def export_to_hdf5(self, path, mode='a', libver='earliest'):
        """Export radioactive decay data to an HDF5 file

        One file per nuclide, as in the ENDF decay sublibrary; every value
        keeps its standard deviation.

        .. versionadded:: 0.16.1

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
            f.attrs['filetype'] = np.bytes_('data_decay')
            f.attrs['version'] = np.array(HDF5_VERSION)
            g = f.create_group(self.nuclide['name'])
            nuc = self.nuclide
            for key in ('atomic_number', 'mass_number', 'isomeric_state',
                        'excited_state', 'excitation_energy', 'mass', 'parity'):
                g.attrs[key] = nuc[key]
            g.attrs['stable'] = int(nuc['stable'])
            g.attrs['spin'] = np.nan if nuc['spin'] is None else nuc['spin']
            g.attrs['half_life'] = _uncertain_array([self.half_life])[0]

            if self.average_energies:
                e = g.create_group('average_energies')
                e.create_dataset('names', data=np.array(
                    list(self.average_energies), dtype=h5py.string_dtype()))
                e.create_dataset('values', data=_uncertain_array(
                    self.average_energies.values()))

            if self.modes:
                m = g.create_group('modes')
                m.create_dataset('modes', data=np.array(
                    [','.join(x.modes) for x in self.modes], dtype=h5py.string_dtype()))
                m.create_dataset('daughter_state',
                                 data=[x._daughter_state for x in self.modes])
                m.create_dataset('energy', data=_uncertain_array(
                    [x.energy for x in self.modes]))
                m.create_dataset('branching_ratio', data=_uncertain_array(
                    [x.branching_ratio for x in self.modes]))

            for i, spectrum in enumerate(self.spectra.values()):
                sg = g.create_group(f'spectrum_{i}')
                sg.attrs['type'] = spectrum['type']
                sg.attrs['continuous_flag'] = spectrum['continuous_flag']
                for key in ('discrete_normalization', 'energy_average',
                            'continuous_normalization'):
                    sg.attrs[key] = _uncertain_array([spectrum[key]])[0]
                if 'discrete' in spectrum:
                    lines = spectrum['discrete']
                    d = sg.create_group('discrete')
                    d.create_dataset('energy', data=_uncertain_array(
                        [x['energy'] for x in lines]))
                    d.create_dataset('intensity', data=_uncertain_array(
                        [x['intensity'] for x in lines]))
                    d.create_dataset('from_mode', data=np.array(
                        [','.join(x['from_mode']) for x in lines],
                        dtype=h5py.string_dtype()))
                    d.create_dataset('transition', data=np.array(
                        [x['type'] or '' for x in lines], dtype=h5py.string_dtype()))
                    for key in self._LINE_COLUMNS:
                        if any(key in x for x in lines):
                            d.create_dataset(key, data=_uncertain_array(
                                [x.get(key) for x in lines]))
                if 'continuous' in spectrum:
                    ci = spectrum['continuous']
                    c = sg.create_group('continuous')
                    c.attrs['from_mode'] = ','.join(ci['from_mode'])
                    ci['probability'].to_hdf5(c, 'probability')
                    if 'covariance' in ci:
                        c.attrs['covariance_lb'] = ci['covariance_lb']
                        c.create_dataset('covariance', data=np.array(
                            ci['covariance'], dtype=float).reshape(-1, 2))

    @classmethod
    def from_hdf5(cls, group_or_filename):
        """Read radioactive decay data from an HDF5 file

        .. versionadded:: 0.16.1

        Parameters
        ----------
        group_or_filename : h5py.Group or str
            HDF5 group containing the decay data. If given as a string, it is
            assumed to be the filename of the HDF5 file, and its first group
            is read.

        Returns
        -------
        openmc.data.Decay
            Radioactive decay data

        """
        group, h5file, _ = _hdf5_group(group_or_filename, ('data_decay',))
        try:
            obj = cls.__new__(cls)
            a = group.attrs
            nuc = obj.nuclide = {'name': group.name.lstrip('/')}
            for key in ('atomic_number', 'mass_number', 'isomeric_state',
                        'excited_state'):
                nuc[key] = int(a[key])
            for key in ('excitation_energy', 'mass', 'parity'):
                nuc[key] = float(a[key])
            nuc['stable'] = bool(a['stable'])
            nuc['spin'] = None if np.isnan(a['spin']) else float(a['spin'])
            obj.half_life = _uncertain_values([a['half_life']])[0]

            obj.average_energies = {}
            if 'average_energies' in group:
                e = group['average_energies']
                names = [n.decode() for n in e['names'][()]]
                obj.average_energies = dict(zip(names, _uncertain_values(e['values'][()])))

            obj.modes = []
            if 'modes' in group:
                m = group['modes']
                energies = _uncertain_values(m['energy'][()])
                ratios = _uncertain_values(m['branching_ratio'][()])
                for modes, state, energy, br in zip(
                        m['modes'][()], m['daughter_state'][()], energies, ratios):
                    obj.modes.append(DecayMode(nuc['name'], modes.decode().split(','),
                                               int(state), energy, br))

            obj.spectra = {}
            i = 0
            while f'spectrum_{i}' in group:
                sg = group[f'spectrum_{i}']
                spectrum = {'type': sg.attrs['type'],
                            'continuous_flag': sg.attrs['continuous_flag']}
                for key in ('discrete_normalization', 'energy_average',
                            'continuous_normalization'):
                    spectrum[key] = _uncertain_values([sg.attrs[key]])[0]
                if 'discrete' in sg:
                    d = sg['discrete']
                    columns = {key: _uncertain_values(d[key][()])
                               for key in ('energy', 'intensity') + cls._LINE_COLUMNS
                               if key in d}
                    lines = []
                    for j, (mode, transition) in enumerate(
                            zip(d['from_mode'][()], d['transition'][()])):
                        line = {'energy': columns['energy'][j],
                                'from_mode': mode.decode().split(','),
                                'type': transition.decode() or None,
                                'intensity': columns['intensity'][j]}
                        for key in cls._LINE_COLUMNS:
                            if key in columns and columns[key][j] is not None:
                                line[key] = columns[key][j]
                        lines.append(line)
                    spectrum['discrete'] = lines
                if 'continuous' in sg:
                    c = sg['continuous']
                    ci = {'probability': Tabulated1D.from_hdf5(c['probability']),
                          'from_mode': c.attrs['from_mode'].split(',')}
                    if 'covariance' in c:
                        ci['covariance_lb'] = int(c.attrs['covariance_lb'])
                        ci['covariance'] = [tuple(x) for x in c['covariance'][()]]
                    spectrum['continuous'] = ci
                obj.spectra[spectrum['type']] = spectrum
                i += 1
        finally:
            if h5file is not None:
                h5file.close()
        return obj

    @cached_property
    def sources(self):
        """Radioactive decay source distributions"""
        sources = {}
        name = self.nuclide['name']
        decay_constant = self.decay_constant.n
        for particle, spectra in self.spectra.items():
            # Set particle type based on 'particle' above
            particle_type = {
                'gamma': 'photon',
                'beta-': 'electron',
                'ec/beta+': 'positron',
                'alpha': 'alpha',
                'n': 'neutron',
                'sf': 'fragment',
                'p': 'proton',
                'e-': 'electron',
                'xray': 'photon',
                'anti-neutrino': 'anti-neutrino',
                'neutrino': 'neutrino',
            }[particle]

            # Create distribution for discrete
            if (spectra['continuous_flag'] in ('discrete', 'both')
                    and spectra['discrete']):
                energies = []
                intensities = []
                for discrete_data in spectra['discrete']:
                    energies.append(discrete_data['energy'].n)
                    intensities.append(discrete_data['intensity'].n)
                energies = np.array(energies)
                intensity = spectra['discrete_normalization'].n
                rates = decay_constant * intensity * np.array(intensities)
                dist_discrete = Discrete(energies, rates)
                sources.setdefault(particle_type, []).append(dist_discrete)

            # Create distribution for continuous
            if spectra['continuous_flag'] in ('continuous', 'both'):
                f = spectra['continuous']['probability']
                if len(f.interpolation) > 1:
                    raise NotImplementedError("Multiple interpolation regions: {name}, {particle}")
                interpolation = INTERPOLATION_SCHEME[f.interpolation[0]]
                if interpolation not in ('histogram', 'linear-linear'):
                    warn(
                        f"Continuous spectra with {interpolation} interpolation "
                        f"({name}, {particle}) encountered.")

                intensity = spectra['continuous_normalization'].n
                rates = decay_constant * intensity * f.y
                dist_continuous = Tabular(f.x, rates, interpolation)
                sources.setdefault(particle_type, []).append(dist_continuous)

        # Combine discrete distributions
        merged_sources = {}
        for particle_type, dist_list in sources.items():
            merged_sources[particle_type] = combine_distributions(
                dist_list, [1.0]*len(dist_list))

        return merged_sources


_DECAY_PHOTON_ENERGY = {}


def decay_photon_energy(nuclide: str) -> Univariate | None:
    """Get photon energy distribution resulting from the decay of a nuclide

    This function relies on data stored in a depletion chain. Before calling it
    for the first time, you need to ensure that a depletion chain has been
    specified in openmc.config['chain_file'].

    .. versionadded:: 0.13.2

    Parameters
    ----------
    nuclide : str
        Name of nuclide, e.g., 'Co58'

    Returns
    -------
    openmc.stats.Univariate or None
        Distribution of energies in [eV] of photons emitted from decay, or None
        if no photon source exists. Note that the probabilities represent
        intensities, given as [Bq/atom] (in other words, decay constants).
    """
    if not _DECAY_PHOTON_ENERGY:
        chain_file = openmc.config.get('chain_file')
        if chain_file is None:
            raise DataError(
                "A depletion chain file must be specified with "
                "openmc.config['chain_file'] in order to load decay data."
            )

        from openmc.deplete import Chain
        chain = Chain.from_xml(chain_file)
        for nuc in chain.nuclides:
            if 'photon' in nuc.sources:
                _DECAY_PHOTON_ENERGY[nuc.name] = nuc.sources['photon']

        # If the chain file contained no sources at all, warn the user
        if not _DECAY_PHOTON_ENERGY:
            warn(f"Chain file '{chain_file}' does not have any decay photon "
                 "sources listed.")

    return _DECAY_PHOTON_ENERGY.get(nuclide)


_DECAY_ENERGY = {}


def decay_energy(nuclide: str):
    """Get decay energy value resulting from the decay of a nuclide

    This function relies on data stored in a depletion chain. Before calling it
    for the first time, you need to ensure that a depletion chain has been
    specified in openmc.config['chain_file'].

    .. versionadded:: 0.13.3

    Parameters
    ----------
    nuclide : str
        Name of nuclide, e.g., 'H3'

    Returns
    -------
    float
        Decay energy of nuclide in [eV]. If the nuclide is stable, a value of
        0.0 is returned.
    """
    if not _DECAY_ENERGY:
        chain_file = openmc.config.get('chain_file')
        if chain_file is None:
            raise DataError(
                "A depletion chain file must be specified with "
                "openmc.config['chain_file'] in order to load decay data."
            )

        from openmc.deplete import Chain
        chain = Chain.from_xml(chain_file)
        for nuc in chain.nuclides:
            if nuc.decay_energy:
                _DECAY_ENERGY[nuc.name] = nuc.decay_energy

        # If the chain file contained no decay energy, warn the user
        if not _DECAY_ENERGY:
            warn(f"Chain file '{chain_file}' does not have any decay energy.")

    return _DECAY_ENERGY.get(nuclide, 0.0)
