"""Neutron-proton elastic scattering from the Nijmegen partial-wave analysis.

The tabulated differential cross sections in ``np_pwa93.txt`` are those of
PWA93 (V. G. J. Stoks, R. A. M. Klomp, M. C. M. Rentmeester, and J. J. de
Swart, Phys. Rev. C 48, 792 (1993)) as computed by NN-OnLine; see
``tools/data/generate_np_pwa93.py``.

"""
from functools import lru_cache
from pathlib import Path

import numpy as np

from openmc.stats import Tabular
from .angle_distribution import AngleDistribution
from .function import Tabulated1D


@lru_cache()
def _table():
    data = np.loadtxt(Path(__file__).with_name('np_pwa93.txt'))
    angles = data[0, 1:]
    energy = data[1:, 0]
    dxs = data[1:, 1:]*1e-3
    # Ascending cosine, with the 1 degree value held to 0 degrees
    mu = np.append(np.cos(np.radians(angles))[::-1], 1.0)
    dxs = np.hstack([dxs[:, ::-1], dxs[:, -1:]])
    mu[0] = -1.0
    return energy, mu, dxs


def np_elastic_xs():
    """Neutron-proton elastic cross section.

    Returns
    -------
    openmc.data.Tabulated1D
        Cross section in b as a function of laboratory energy in eV,
        linear-linear between the tabulated energies

    """
    energy, mu, dxs = _table()
    xs = 2*np.pi*np.trapezoid(dxs, mu, axis=1)
    return Tabulated1D(energy*1e6, xs)


def np_elastic_angle():
    """Center-of-mass angular distribution of neutron-proton elastic scattering.

    Returns
    -------
    openmc.data.AngleDistribution
        Distribution of the cosine of the neutron's center-of-mass
        scattering angle, linear-linear in cosine

    """
    energy, mu, dxs = _table()
    dists = []
    for row in dxs:
        p = row / np.trapezoid(row, mu)
        dists.append(Tabular(mu, p, 'linear-linear'))
    return AngleDistribution(energy*1e6, dists)
