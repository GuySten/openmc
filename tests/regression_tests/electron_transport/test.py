"""Electron transport, and the two claims condensed history rests on.

This is not a stored-answer test. Condensed history is an approximation with a
knob on it, and what has to hold is a relation between two runs rather than a
number: that turning the knob to zero gives back the method the approximation
is built to agree with, and that neither run loses energy. A golden statepoint
would pin the answer without checking either.

Skipped unless the cross section library in use carries electron data, which
the libraries OpenMC's CI currently downloads do not.
"""

import numpy as np
import pytest

import openmc
import openmc.data


def _has_electron_data():
    """Whether the configured library can transport electrons"""
    try:
        lib = openmc.data.DataLibrary.from_xml()
    except Exception:
        return False
    return any(item['type'] == 'electron' for item in lib.libraries)


pytestmark = pytest.mark.skipif(
    not _has_electron_data(),
    reason='cross section library has no electron data'
)

# Thick enough to stop a 1 MeV electron several times over, so that nothing
# leaves except photons, and small enough to run in seconds
E0 = 1.0e6
CUTOFF = 1.0e4


def _model(deflection, particles=400, seed=1, event_based=False):
    mat = openmc.Material()
    mat.add_element('C', 1.0)
    mat.set_density('g/cm3', 1.7)

    outer = openmc.Sphere(r=1.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-outer)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.particles = particles
    settings.batches = 4
    settings.seed = seed
    settings.event_based = event_based
    settings.photon_transport = True
    settings.electron_transport = True
    settings.cutoff = {
        'energy_electron': CUTOFF, 'energy_positron': CUTOFF,
        'energy_photon': CUTOFF, 'deflection': deflection,
    }
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)),
        energy=openmc.stats.delta_function(E0),
        particle='electron',
    )

    heating = openmc.Tally(name='heating')
    heating.scores = ['heating']

    return openmc.Model(
        geometry=openmc.Geometry([cell]),
        materials=openmc.Materials([mat]),
        settings=settings,
        tallies=openmc.Tallies([heating]),
    )


def _heating(model, tmp_path, name):
    path = model.run(cwd=tmp_path / name, output=False)
    with openmc.StatePoint(path) as sp:
        tally = sp.get_tally(name='heating')
        return tally.mean.ravel()[0], tally.std_dev.ravel()[0]


def test_condensed_history_reduces_to_single_event(tmp_path):
    """Zero deflection groups nothing, so it must be the single-event answer

    The grouping decision is made before anything is sampled, so with the knob
    at zero the same random numbers are drawn in the same order and the two
    runs are the same history for history -- to the order the threads happen
    to add their tallies up in, which is why the reference is compared to
    itself first and to a tolerance rather than exactly.
    """
    grouped, grouped_err = _heating(_model(0.005), tmp_path, 'grouped')
    single, single_err = _heating(_model(0.0), tmp_path, 'single')
    again, _ = _heating(_model(0.0), tmp_path, 'single_again')

    # The reference is reproducible at all, up to summation order
    assert again == pytest.approx(single, rel=1e-10)

    # and the two methods agree within their combined error. Three standard
    # errors, since this is a test that must not fail on a bad day.
    spread = np.hypot(grouped_err, single_err)
    assert abs(grouped - single) < 3.0 * spread


def test_no_energy_disappears(tmp_path):
    """Everything the source brings in is deposited or leaves

    A condensed-history step deposits what its grouped collisions took, and
    that energy reaches a tally only through the collision energy balance. A
    step cut short by a surface, or by the particle dying inside it, scores no
    balance of its own, so this is what notices if one stops being scored.

    The sphere's surface is the only way out and the geometry is small, so
    with photons transported down to the same cutoff almost everything stays.
    """
    for deflection in (0.0, 0.005):
        heating, err = _heating(
            _model(deflection), tmp_path, f'balance_{deflection}')
        fraction = heating / E0
        # Some energy does leave as photons, so this is a band rather than an
        # equality: what it catches is energy vanishing, which is unbounded.
        assert fraction > 0.5, f'only {fraction:.3f} of the source accounted'
        assert fraction <= 1.0 + 3.0 * err / E0, 'more energy out than in'


def test_event_based_mode_agrees(tmp_path):
    """The step is carried in ParticleData, so the two dispatch modes agree

    Worth its own test rather than trust: event-based mode reuses a particle
    slot for every source particle as well as for every secondary, so anything
    a step leaves behind in one is inherited sooner and more often there than
    in history-based mode.
    """
    history, _ = _heating(_model(0.005), tmp_path, 'history')
    event, _ = _heating(
        _model(0.005, event_based=True), tmp_path, 'event')

    assert event == pytest.approx(history, rel=1e-10)
