import numpy as np
import pytest

from openmc.data.np_elastic import np_elastic_angle, np_elastic_xs


@pytest.mark.parametrize('energy, total', [
    # Measured np total cross sections (no inelastic channels below the pion
    # threshold), Abfalterer et al., Phys. Rev. C 63, 044608 (2001),
    # EXFOR 13753032, averaged over +-5% in energy
    (20.0, 0.4812), (50.0, 0.1669), (100.0, 0.0760), (150.0, 0.0518)
])
def test_np_elastic_xs(energy, total):
    assert np_elastic_xs()(energy*1e6) == pytest.approx(total, rel=0.02)


def test_np_elastic_angle():
    angle = np_elastic_angle()
    assert angle.energy[0] == 20e6
    assert angle.energy[-1] == 200e6
    for dist in angle.mu:
        assert dist.x[0] == -1.0
        assert dist.x[-1] == 1.0
        assert np.all(np.diff(dist.x) > 0)
        assert np.trapezoid(dist.p, dist.x) == pytest.approx(1.0)
    # The charge-exchange peak makes backward neutron scattering more likely
    # than forward scattering, increasingly so with energy
    at = lambda d, mu: np.interp(mu, d.x, d.p)
    ratios = [at(d, -1.0)/at(d, 0.98) for d in angle.mu]
    assert ratios[0] > 1.0
    assert ratios[-1] > ratios[0]
