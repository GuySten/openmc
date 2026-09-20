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


# The Landau most probable loss, and what a thin foil is for
# ---------------------------------------------------------
# Landau's formula gives the most probable COLLISION loss of a fast charged
# particle in a foil thin enough that its energy hardly changes:
#
#   Dp = xi [ ln(2 m c^2 beta^2 gamma^2 / I) + ln(xi / I) + j - beta^2 - delta ]
#
# with xi = (K/2)(Z/A) x / beta^2. It is a property of the medium and of the
# thickness alone, so it tests the transport against something outside the
# code -- and it tests the SHAPE of the energy-loss distribution rather than
# its mean, which is what a mean stopping power cannot check. A hard transfer
# spectrum short in its tail pays for it in the middle, keeps the mean, and
# reports the peak too far out: that is exactly the failure this catches.
CU_RHO = 8.92
CU_X = 1.372                          # g/cm2, ~12 grouped steps at 16 MeV
CU_E0 = 16.1e6
CU_STERNHEIMER = (-0.0254, 3.2792, -4.4190, 0.14339, 2.9044)


def _landau_mode(Z, A, x, I, sternheimer, E0):
    """Most probable collision loss in [eV], Landau's formula"""
    mc2 = 0.51099895e6
    gamma = E0 / mc2 + 1.0
    beta_sq = 1.0 - 1.0 / gamma**2
    xi = 0.1535 * (Z / A) * x / beta_sq * 1e6
    X0, X1, C, a, m = sternheimer
    X = np.log10(np.sqrt(gamma * gamma - 1.0))
    delta = 4.6052 * X + C + (a * (X1 - X) ** m if X < X1 else 0.0)
    return xi * (np.log(2 * mc2 * beta_sq * gamma**2 / I) + np.log(xi / I)
                 + 0.200 - beta_sq - delta)


def _foil_model(deflection, energy_loss, particles=30000, seed=11):
    """Copper foil thin enough to stay in the Landau regime"""
    openmc.reset_auto_ids()
    cu = openmc.Material()
    cu.add_element('Cu', 1.0)
    cu.set_density('g/cm3', CU_RHO)
    thickness = CU_X / CU_RHO
    box = openmc.model.RectangularParallelepiped(
        -2.0, 2.0, -2.0, 2.0, 0.0, thickness, boundary_type='vacuum')
    cell = openmc.Cell(fill=cu, region=-box)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.batches = 10
    settings.particles = particles // 10
    settings.seed = seed
    settings.photon_transport = True
    settings.electron_transport = True
    settings.cutoff = {'energy_photon': 1.0e3, 'energy_electron': 1.0e3,
                       'energy_positron': 1.0e3, 'deflection': deflection,
                       'energy_loss': energy_loss}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 1.0e-6)),
        angle=openmc.stats.Monodirectional((0.0, 0.0, 1.0)),
        energy=openmc.stats.delta_function(CU_E0), particle='electron')

    edges = np.linspace(0.85 * CU_E0, CU_E0, 121)
    spectrum = openmc.Tally(name='spectrum')
    spectrum.filters = [openmc.ParticleFilter(['electron']),
                        openmc.SurfaceFilter([box.zmax]),
                        openmc.EnergyFilter(edges)]
    spectrum.scores = ['current']
    return openmc.Model(openmc.Geometry([cell]), openmc.Materials([cu]),
                        settings, openmc.Tallies([spectrum])), edges


def _most_probable_loss(deflection, energy_loss, tmp_path, name):
    """Peak of the energy-loss distribution in [eV], by parabola"""
    model, edges = _foil_model(deflection, energy_loss)
    with model.run(cwd=tmp_path / name, output=False) as path:
        with openmc.StatePoint(path) as sp:
            counts = sp.get_tally(name='spectrum').mean.ravel()

    loss = CU_E0 - 0.5 * (edges[:-1] + edges[1:])
    peak = int(np.argmax(counts))
    if 0 < peak < len(counts) - 1:
        c2, c1, _ = np.polyfit(loss[peak - 1:peak + 2],
                               counts[peak - 1:peak + 2], 2)
        if c2 < 0.0:
            return -c1 / (2.0 * c2)
    return loss[peak]


def test_landau_most_probable_loss(tmp_path):
    """The peak of the energy-loss distribution sits where Landau says

    Not an equality. Landau's is the collision loss alone, and a real foil
    also radiates, which can only move the peak up; the foil is thick enough
    that beta changes a little across it, which the formula assumes it does
    not. Measured, those are worth five per cent here and seven per cent for
    1 MeV electrons in aluminium, so the band is generous on the high side and
    tight on the low, where nothing physical can take the peak.

    What it catches is the peak in the wrong place by more than that, which is
    what a transfer spectrum with too few large transfers reports: before the
    hard channel carried max(evaluated, free), this came out at 1.16 times the
    Landau value.
    """
    theory = _landau_mode(29, 63.546, CU_X, 322.0, CU_STERNHEIMER, CU_E0)
    measured = _most_probable_loss(0.01, 0.01, tmp_path, 'landau')

    ratio = measured / theory
    assert ratio > 0.98, (
        f'peak at {ratio:.3f} of the Landau loss; nothing physical puts it '
        'below, so the collision spectrum is too soft')
    assert ratio < 1.12, (
        f'peak at {ratio:.3f} of the Landau loss; too far out for '
        'bremsstrahlung to explain, so the transfer spectrum is short in its '
        'tail')


def test_most_probable_loss_survives_the_step(tmp_path):
    """The peak does not move when the step bounds are opened up

    Sharper than the comparison with Landau, and free of its assumptions: the
    grouped step is an approximation with a knob, and the distribution it
    produces must not depend on where the knob is set. A step that swallows
    too much of the spectrum shows up here first.
    """
    fine = _most_probable_loss(0.01, 0.01, tmp_path, 'fine')
    coarse = _most_probable_loss(0.05, 0.05, tmp_path, 'coarse')

    assert coarse == pytest.approx(fine, rel=0.04), (
        f'peak moved from {fine:.4g} to {coarse:.4g} eV when the step bounds '
        'were opened up five-fold')


def _k_xray_yield(deflection, tmp_path, name, particles=30000):
    """Cu K x-rays leaving a thin foil, per incident electron"""
    openmc.reset_auto_ids()
    cu = openmc.Material()
    cu.add_element('Cu', 1.0)
    cu.set_density('g/cm3', CU_RHO)
    box = openmc.model.RectangularParallelepiped(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.002, boundary_type='vacuum')
    cell = openmc.Cell(fill=cu, region=-box)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.batches = 10
    settings.particles = particles // 10
    settings.seed = 5
    settings.photon_transport = True
    settings.electron_transport = True
    settings.cutoff = {'energy_photon': 1.0e3, 'energy_electron': 1.0e3,
                       'energy_positron': 1.0e3, 'deflection': deflection,
                       'energy_loss': 0.01}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 1.0e-4)),
        angle=openmc.stats.Monodirectional((0.0, 0.0, 1.0)),
        energy=openmc.stats.delta_function(1.0e6), particle='electron')

    # Cu K-alpha is 8.05 keV and K-beta 8.90 keV; nothing else lands here
    lines = openmc.Tally(name='k')
    lines.filters = [openmc.ParticleFilter(['photon']),
                     openmc.SurfaceFilter([box.zmax]),
                     openmc.EnergyFilter([7.5e3, 9.5e3])]
    lines.scores = ['current']
    model = openmc.Model(openmc.Geometry([cell]), openmc.Materials([cu]),
                         settings, openmc.Tallies([lines]))
    with model.run(cwd=tmp_path / name, output=False) as path:
        with openmc.StatePoint(path) as sp:
            tally = sp.get_tally(name='k')
            return tally.mean.ravel()[0], tally.std_dev.ravel()[0]


def test_inner_shell_vacancies_survive_grouping(tmp_path):
    """Grouping does not cost the K vacancies that make the x-rays

    The K-shell ionization cross section is mostly distant: the free binary
    cross section accounts for under a third of it in copper, and under a
    fifth in lead. A hard channel built from the free cross section alone
    therefore keeps the energy -- the stopping power is held to ICRU 37 either
    way -- while quietly losing three quarters of the vacancies, and with them
    the characteristic x rays. Single-event transport never uses that channel,
    so it is the control.

    The tally is the two Cu K lines, which nothing else in the problem makes.
    """
    grouped, grouped_err = _k_xray_yield(0.01, tmp_path, 'k_grouped')
    single, single_err = _k_xray_yield(0.0, tmp_path, 'k_single')

    assert grouped > 0.0, 'no K x-rays at all under condensed history'
    spread = np.hypot(grouped_err, single_err)
    assert abs(grouped - single) < 3.0 * spread, (
        f'K x-ray yield {grouped:.4e} grouped against {single:.4e} '
        'single-event: grouping is losing inner-shell vacancies')


# What the settings mean together
# -------------------------------
# The charged-particle settings arrived one at a time and can be combined
# freely, including with the settings that predate them. What one means for
# another is decided in one place, and these are the combinations that place
# has to get right. A setting that does not apply is ignored with a warning
# rather than refused: a script sweeping a parameter should not fail on the
# cases where the parameter does not bite.
def _combination_model(tmp_path, name, electron_transport=True, **kwargs):
    openmc.reset_auto_ids()
    mat = openmc.Material()
    mat.add_element('C', 1.0)
    mat.set_density('g/cm3', 1.7)
    outer = openmc.Sphere(r=1.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-outer)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.particles = 20
    settings.batches = 2
    settings.seed = 3
    settings.photon_transport = True
    settings.electron_transport = electron_transport
    settings.cutoff = {'energy_electron': CUTOFF, 'energy_positron': CUTOFF,
                       'energy_photon': CUTOFF}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0.0, 0.0, 0.0)),
        energy=openmc.stats.delta_function(E0), particle='electron')
    for key, value in kwargs.items():
        if key == 'cutoff':
            settings.cutoff = {**settings.cutoff, **value}
        else:
            setattr(settings, key, value)

    heating = openmc.Tally(name='heating')
    heating.scores = ['heating']
    return openmc.Model(openmc.Geometry([cell]), openmc.Materials([mat]),
                        settings, openmc.Tallies([heating]))


def _run_capturing(model, tmp_path, name):
    """Run and give back (combined output, heating) -- heating None if it died"""
    import subprocess
    cwd = tmp_path / name
    cwd.mkdir(parents=True, exist_ok=True)
    model.export_to_model_xml(cwd / 'model.xml')
    proc = subprocess.run(['openmc'], cwd=cwd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return out, None
    sp = list(cwd.glob('statepoint.*.h5'))
    with openmc.StatePoint(sp[0]) as s:
        return out, s.get_tally(name='heating').mean.ravel()[0]


def test_settings_that_do_not_apply_are_ignored(tmp_path):
    """Without electron transport the charged-particle knobs have no target

    Each is ignored with a word about it, and the answer is the one the run
    would have given without it. Refusing to run would be worse: these are
    quality and variance-reduction knobs, and a sweep over one should not
    fail on the cases where nothing charged is transported.
    """
    plain, plain_heat = _run_capturing(
        _combination_model(tmp_path, 'plain', electron_transport=False),
        tmp_path, 'plain')
    assert plain_heat is not None, plain

    for name, kwargs, phrase in [
        ('split', {'bremsstrahlung_split': 3}, 'Bremsstrahlung splitting'),
        ('density', {'density_effect': False}, 'density-effect correction'),
        ('deflect', {'cutoff': {'deflection': 0.05}}, 'deflection cutoff'),
        ('loss', {'cutoff': {'energy_loss': 0.05}}, 'energy loss cutoff'),
    ]:
        out, heat = _run_capturing(
            _combination_model(tmp_path, name, electron_transport=False,
                               **kwargs), tmp_path, name)
        assert heat is not None, out
        assert 'ignored' in out and phrase in out, out
        assert heat == pytest.approx(plain_heat, rel=1e-10), (
            f'{name} changed the answer although it does not apply')


def test_ttb_gives_way_to_electron_transport(tmp_path):
    """Asking for both says so, and transports rather than approximating"""
    out, heat = _run_capturing(
        _combination_model(tmp_path, 'ttb', electron_treatment='ttb'),
        tmp_path, 'ttb')
    assert heat is not None, out
    assert 'ttb' in out and 'ignored' in out, out

    _, led = _run_capturing(
        _combination_model(tmp_path, 'led', electron_treatment='led'),
        tmp_path, 'led')
    assert heat == pytest.approx(led, rel=1e-10)


def test_electron_transport_turns_photon_transport_on(tmp_path):
    """The data it needs is loaded with the photons, so it says so and does it"""
    model = _combination_model(tmp_path, 'nophoton')
    model.settings.photon_transport = False
    out, heat = _run_capturing(model, tmp_path, 'nophoton')
    assert heat is not None, out
    assert 'requires photon transport' in out, out


def test_charged_flux_with_a_collision_estimator_is_refused(tmp_path):
    """Zero for every bin is not an answer a tally may return in silence"""
    for estimator in ('analog', 'collision'):
        model = _combination_model(tmp_path, f'flux_{estimator}')
        flux = openmc.Tally(name='flux')
        flux.filters = [openmc.ParticleFilter(['electron'])]
        flux.scores = ['flux']
        flux.estimator = estimator
        model.tallies.append(flux)

        out, heat = _run_capturing(model, tmp_path, f'flux_{estimator}')
        assert heat is None, 'the run should have stopped'
        assert 'tracklength' in out, out

    # and a tracklength estimator of the same tally runs
    model = _combination_model(tmp_path, 'flux_tl')
    flux = openmc.Tally(name='flux')
    flux.filters = [openmc.ParticleFilter(['electron'])]
    flux.scores = ['flux']
    flux.estimator = 'tracklength'
    model.tallies.append(flux)
    out, heat = _run_capturing(model, tmp_path, 'flux_tl')
    assert heat is not None, out
