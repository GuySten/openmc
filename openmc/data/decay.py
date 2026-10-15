from collections.abc import Iterable
from functools import cached_property
from io import StringIO
from math import log
from warnings import warn

import numpy as np
from uncertainties import ufloat, UFloat

import openmc
import openmc.checkvalue as cv
from openmc.exceptions import DataError
from openmc.mixin import EqualityMixin
from openmc.stats import (BetaSpectrum, Discrete, Mixture, Tabular, Univariate,
                          combine_distributions)
from .data import gnds_name, zam
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


def _energy_integral(f):
    """Integral of energy times a tabulated spectrum

    Parameters
    ----------
    f : openmc.data.Tabulated1D
        Spectrum with histogram or linear-linear interpolation

    Returns
    -------
    float
        Integral of E*f(E)

    """
    x, y = f.x, f.y
    if INTERPOLATION_SCHEME[f.interpolation[0]] == 'histogram':
        return np.sum(0.5*(x[1:] + x[:-1]) * y[:-1] * np.diff(x))
    dx = np.diff(x)
    return np.sum(dx/6.0 * ((2*x[:-1] + x[1:])*y[:-1] + (x[:-1] + 2*x[1:])*y[1:]))


# Electron mass energy equivalent in [eV]
_MASS_ELECTRON_EV = 0.51099895000e6

# Tolerance in [eV] for identifying the transition that feeds the daughter
# state of a decay mode from its Q-value
_Q_VALUE_TOLERANCE = 5.0e3


def _spin_parity(nuclide):
    """Spin and parity of a nuclide, or None if either is unknown"""
    spin = nuclide.get('spin')
    parity = nuclide.get('parity')
    # ENDF-102 specifies that unknown spin should be reported as -77.777
    if spin is None or spin < 0.0 or parity not in (-1.0, 1.0):
        return None
    return spin, parity


def _transition_shape(initial, final):
    """Forbiddenness and uniqueness of a beta transition

    Parameters
    ----------
    initial : tuple of float
        Spin and parity of the initial state
    final : tuple of float
        Spin and parity of the final state

    Returns
    -------
    int
        Order of forbiddenness
    bool
        Whether the transition is unique forbidden

    """
    delta_j = round(abs(initial[0] - final[0]))
    delta_pi = round(initial[1] * final[1])
    if delta_j <= 1:
        # Allowed or non-unique first forbidden
        return (0, False) if delta_pi == 1 else (1, False)
    if delta_pi == (-1)**(delta_j - 1):
        # Unique transition of order n, with delta J = n + 1
        return delta_j - 1, True
    return delta_j, False


def _beta_spectra(endpoints, rates, Z, A, positron, mean_energy, shapes=None):
    """Spectrum of beta particles from discrete transitions

    Each transition is given a :class:`openmc.stats.BetaSpectrum`, with the
    shape factor of unique forbidden transitions where indicated and the
    allowed shape otherwise. A correction factor :math:`e^{aT}` shared by all
    transitions is applied such that the mean energy of the spectrum is
    preserved. Among all spectra with the given mean energy and transition
    intensities, this is the one closest to the uncorrected shapes in relative
    entropy.

    Parameters
    ----------
    endpoints : sequence of float
        Endpoint (maximum kinetic) energies of the transitions in [eV]
    rates : sequence of float
        Emission rate of each transition
    Z : int
        Atomic number of the daughter nucleus
    A : int
        Mass number of the daughter nucleus
    positron : bool
        Whether the beta particles are positrons
    mean_energy : float
        Mean kinetic energy of the beta particles in [eV] to preserve
    shapes : sequence of tuple, optional
        Order of forbiddenness and whether the transition is unique for each
        transition. If not given, all transitions have the allowed shape.

    Returns
    -------
    openmc.stats.Mixture
        Mixture of beta spectra whose integral is the total emission rate

    """
    e_max = max(endpoints)
    rates = np.asarray(rates, dtype=float)

    # The correction multiplies the density by exp(a*(T - endpoint)), so the
    # density of each transition is evaluated once at the quadrature points
    if shapes is None:
        shapes = [(0, False)]*len(endpoints)
    # Non-unique forbidden transitions have the allowed shape
    shapes = [(n, True) if unique else (0, False) for n, unique in shapes]
    quadrature = []
    for e0, (n, unique) in zip(endpoints, shapes):
        spectrum = BetaSpectrum(e0, Z, A, positron, forbiddenness=n,
                                unique=unique)
        T, weights = spectrum._quadrature()
        quadrature.append((T, weights * spectrum._shape(T), e0))

    def mean(b):
        total = 0.0
        for rate, (T, f, e0) in zip(rates, quadrature):
            g = f * np.exp(b / e_max * (T - e0))
            total += rate * np.sum(g * T) / np.sum(g)
        return total / rates.sum()

    # Find the correction that preserves the mean energy, which increases
    # monotonically with b. The correction is bounded to keep the spectra
    # meaningful for evaluations whose mean energy is inconsistent with the
    # endpoints.
    b = 0.0
    if mean_energy > 0.0:
        lo, hi = -100.0, 100.0
        if mean_energy <= mean(lo):
            b = lo
        elif mean_energy >= mean(hi):
            b = hi
        else:
            while hi - lo > 1e-9:
                b = 0.5*(lo + hi)
                if mean(b) < mean_energy:
                    lo = b
                else:
                    hi = b
    spectra = [BetaSpectrum(e0, Z, A, positron, b / e_max, forbiddenness=n,
                            unique=unique)
               for e0, (n, unique) in zip(endpoints, shapes)]
    return Mixture(list(rates), spectra)


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
    independent : list of dict
        Independent yields for each tabulated energy. Each item in the list is a
        dictionary whose keys are nuclide names and values are independent
        yields. The i-th dictionary corresponds to the i-th incident neutron
        energy.
    nuclide : dict
        Properties of the fissioning nuclide.

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
            for i in range(n_energy):
                # Determine i-th energy and number of products
                items, values = get_list_record(file_obj)
                energies[i] = items[0]
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

            return energies, data

        ev = as_evaluation(ev_or_filename)

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
            self.energies, self.independent = get_yields(file_obj)

        # Read cumulative yields (MF=8, MT=459)
        if (8, 459) in ev.section:
            file_obj = StringIO(ev.section[8, 459])
            energies, self.cumulative = get_yields(file_obj)
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
        'excited_state', 'mass', 'stable', 'spin', and 'parity'.
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
                        ci['covariance'] = zip(values[0::2], values[1::2])

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

    @cached_property
    def sources(self):
        """Radioactive decay source distributions"""
        return self.get_sources()

    def _final_states(self, transitions):
        """Daughter state fed by each discrete beta transition

        The transition with an energy closest to the Q-value of a decay mode
        feeds the daughter state of that mode, which is the ground state or an
        isomer. Other transitions feed excited states.

        Parameters
        ----------
        transitions : list of dict
            Discrete transitions of a beta- or EC/beta+ spectrum

        Returns
        -------
        list of str or None
            Name of the daughter state fed by each transition, if known

        """
        states = [None]*len(transitions)
        for mode in self.modes:
            q_value = mode.energy.n
            distance = [abs(d['energy'].n - q_value)
                        if d['from_mode'] == mode.modes else np.inf
                        for d in transitions]
            if not distance:
                continue
            i = int(np.argmin(distance))
            # Allow for rounding of the Q-value and transition energy
            if distance[i] <= _Q_VALUE_TOLERANCE:
                states[i] = mode.daughter
        return states

    def get_sources(self, decay_data=None):
        """Radioactive decay source distributions

        Beta particles from discrete transitions are given the shape of a
        unique forbidden transition where the transition is known to be one,
        and the allowed shape otherwise, with a correction that preserves the
        evaluated average energy. Since the transition types in ENDF-6
        evaluations give only the order of forbiddenness, unique transitions
        are identified from the spins and parities of the initial and final
        states. These are known for transitions to the ground state or an
        isomer of the daughter nuclide when decay data for the daughter state
        is given. Non-unique forbidden transitions, whose shape depends on
        nuclear matrix elements, are given the allowed shape.

        .. versionadded:: 0.16.1

        Parameters
        ----------
        decay_data : Mapping of str to openmc.data.Decay, optional
            Decay data for daughter states indexed by name, e.g., 'Y90' or
            'Ba137_m1'

        Returns
        -------
        dict
            Distribution of the energy of each type of particle emitted, where
            the probabilities are emission rates in [Bq/atom]

        """
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
                    and spectra['discrete']
                    and particle in ('beta-', 'ec/beta+')):
                # Discrete beta transitions are given by their endpoint
                # energy (for EC/beta+, the electron capture decay energy,
                # which includes the rest mass of the positron and of the
                # electron it annihilates with). For EC/beta+, the intensity
                # includes electron capture and the positron intensity is
                # given separately (RIS, see ENDF-102 Section 8.3). Each
                # transition is given a beta spectrum, corrected so that the
                # evaluated average energy of the spectrum is preserved.
                positron = particle == 'ec/beta+'
                key = 'positron_intensity' if positron else 'intensity'
                normalization = spectra['discrete_normalization'].n
                endpoints = []
                rates = []
                shapes = []
                initial = _spin_parity(self.nuclide)
                states = self._final_states(spectra['discrete'])
                for d, state in zip(spectra['discrete'], states):
                    endpoint = d['energy'].n
                    if positron:
                        endpoint -= 2.0*_MASS_ELECTRON_EV
                    rate = decay_constant * normalization * d[key].n
                    if endpoint > 0.0 and rate > 0.0:
                        endpoints.append(endpoint)
                        rates.append(rate)
                        final = None
                        if decay_data is not None and state in decay_data:
                            final = _spin_parity(decay_data[state].nuclide)
                        if initial is None or final is None:
                            shapes.append((0, False))
                        else:
                            shapes.append(_transition_shape(initial, final))
                energy = spectra['energy_average'].n
                if spectra['continuous_flag'] == 'both':
                    # The average energy includes the continuous part
                    f = spectra['continuous']['probability']
                    energy -= spectra['continuous_normalization'].n * \
                        _energy_integral(f)
                if endpoints:
                    Z = self.nuclide['atomic_number']
                    A = self.nuclide['mass_number']
                    dist = _beta_spectra(
                        endpoints, rates, Z - 1 if positron else Z + 1, A,
                        positron, energy * decay_constant / sum(rates),
                        shapes)
                    sources.setdefault(particle_type, []).append(dist)
            elif (spectra['continuous_flag'] in ('discrete', 'both')
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
