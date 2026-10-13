"""Spherical optical-model calculations for neutrons above the evaluated range.

The potential is the Koning-Delaroche global neutron potential (A. J. Koning
and J. P. Delaroche, Nucl. Phys. A 713, 231 (2003)). The radial Schrödinger
equation is solved partial wave by partial wave with Numerov integration and
matched to Riccati-Bessel functions (no Coulomb interaction for neutrons).
Kinematics are relativistic.

"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import spherical_jn, spherical_yn, lpmv

from .data import ATOMIC_SYMBOL, NEUTRON_MASS
from .talys import adjustment_factor

#: Reduced Planck constant times speed of light in MeV-fm
HBARC = 197.3269804
#: Atomic mass unit in MeV/c^2
AMU_MEV = 931.49410242
#: Neutron mass in MeV/c^2
NEUTRON_MASS_MEV = NEUTRON_MASS * AMU_MEV
#: Square of the pion Compton wavelength in fm^2 used for spin-orbit terms
PION_COMPTON_SQ = 2.0


@dataclass
class OpticalPotential:
    """Energy-dependent depths and fixed geometry of a nucleon potential.

    Depths are in MeV and radii/diffusenesses in fm. Radii are absolute,
    i.e. already multiplied by :math:`A^{1/3}`.

    """
    vv: float
    wv: float
    wd: float
    vso: float
    wso: float
    rv: float
    av: float
    rw: float
    aw: float
    rd: float
    ad: float
    rso: float
    aso: float
    rwso: float = None
    awso: float = None

    def __post_init__(self):
        if self.rwso is None:
            self.rwso = self.rso
        if self.awso is None:
            self.awso = self.aso

    def __call__(self, r, ls):
        """Evaluate the complex potential in MeV.

        Parameters
        ----------
        r : numpy.ndarray
            Radii in fm
        ls : float
            Eigenvalue of :math:`\\mathbf{l}\\cdot\\boldsymbol{\\sigma}`

        """
        fv = _woods_saxon(r, self.rv, self.av)
        fw = _woods_saxon(r, self.rw, self.aw)
        # 4 a d/dr f is negative and peaks at -1 at r = R
        dfd = 4.0 * self.ad * _woods_saxon_deriv(r, self.rd, self.ad)
        dfso = _woods_saxon_deriv(r, self.rso, self.aso) / r
        dfwso = _woods_saxon_deriv(r, self.rwso, self.awso) / r
        return (-self.vv * fv - 1j * (self.wv * fw - self.wd * dfd)
                + PION_COMPTON_SQ * (self.vso*dfso + 1j*self.wso*dfwso) * ls)

    @property
    def radius(self):
        """Radius beyond which the nuclear potential is negligible"""
        return max(self.rv + 12*self.av, self.rw + 12*self.aw,
                   self.rd + 12*self.ad, self.rso + 12*self.aso,
                   self.rwso + 12*self.awso)


def _woods_saxon(r, R, a):
    return 1.0 / (1.0 + np.exp((r - R) / a))


def _woods_saxon_deriv(r, R, a):
    x = np.exp(-np.abs(r - R) / a)
    return -x / (a * (1.0 + x)**2)


#: Koning-Delaroche global neutron parameters. Keys follow the
#: notation of the paper; ``*_A`` multiplies the mass number and ``*_asym``
#: multiplies :math:`(N-Z)/A`.
KD03 = {
    'v1_0': 59.30, 'v1_asym': 21.0, 'v1_A': 0.024,
    'v2_0': 0.007228, 'v2_A': 1.48e-6, 'v3_0': 1.994e-5, 'v3_A': 2.0e-8,
    'v4_0': 7.0e-9,
    'rv_0': 1.3039, 'rv_A': 0.4054, 'av_0': 0.6778, 'av_A': 1.487e-4,
    'w1_0': 12.195, 'w1_A': 0.0167, 'w2_0': 73.55, 'w2_A': 0.0795,
    'd1_0': 16.0, 'd1_asym': 16.0, 'd2_0': 0.0180, 'd2_A': 0.003802,
    'd2_A2': 8.0, 'd2_A3': 156.0, 'd3_0': 11.5,
    'rd_0': 1.3424, 'rd_A': 0.01585, 'ad_0': 0.5446, 'ad_A': 1.656e-4,
    'vso1_0': 5.922, 'vso1_A': 0.0030, 'vso2_0': 0.0040,
    'wso1_0': -3.1, 'wso2_0': 160.0,
    'rso_0': 1.1854, 'rso_A': 0.647, 'aso_0': 0.59,
    'ef_0': -11.2814, 'ef_A': 0.02646,
}

def global_parameters(Z, A, parameters=KD03):
    """Koning-Delaroche global parameters for one nucleus.

    Parameters
    ----------
    Z : int
        Atomic number of the target
    A : int
        Mass number of the target
    parameters : dict
        Global parameters, e.g. :data:`KD03`

    Returns
    -------
    dict
        Parameters for :func:`potential` in the form of a nucleus-specific
        set, with radii in units of :math:`A^{1/3}` fm

    """
    p = parameters
    asym = (A - 2*Z) / A
    a13 = A**(1/3)
    return {
        'ef': p['ef_0'] + p['ef_A']*A,
        'rv': p['rv_0'] - p['rv_A']/a13, 'av': p['av_0'] - p['av_A']*A,
        'v1': p['v1_0'] - p['v1_asym']*asym - p['v1_A']*A,
        'v2': p['v2_0'] - p['v2_A']*A, 'v3': p['v3_0'] - p['v3_A']*A,
        'v4': p['v4_0'],
        'w1': p['w1_0'] + p['w1_A']*A, 'w2': p['w2_0'] + p['w2_A']*A,
        'rd': p['rd_0'] - p['rd_A']*a13, 'ad': p['ad_0'] - p['ad_A']*A,
        'd1': p['d1_0'] - p['d1_asym']*asym,
        'd2': p['d2_0'] + p['d2_A']/(
            1.0 + np.exp((A - p['d2_A3'])/p['d2_A2'])),
        'd3': p['d3_0'],
        'rso': p['rso_0'] - p['rso_A']/a13, 'aso': p['aso_0'],
        'vso1': p['vso1_0'] + p['vso1_A']*A, 'vso2': p['vso2_0'],
        'wso1': p['wso1_0'], 'wso2': p['wso2_0'],
    }


def potential(params, A, energy, factors=None):
    """Optical potential from Koning-Delaroche-form parameters.

    Implements the Koning-Delaroche energy dependence as TALYS does
    (``opticalnp.f90``), including its multiplicative adjustment factors.

    Parameters
    ----------
    params : dict
        Parameters as returned by :func:`global_parameters` or
        :func:`openmc.data.talys.read_omp_file`
    A : int
        Mass number of the target
    energy : float
        Incident neutron energy in the laboratory system in MeV
    factors : dict, optional
        Multiplication factors keyed by the names in
        :data:`openmc.data.talys.ADJUST_KEYS`

    Returns
    -------
    OpticalPotential
        Potential depths and geometry

    """
    F = {} if factors is None else factors
    p = params
    a13 = A**(1/3)
    f = energy - p['ef']
    v1 = F.get('v1', 1.0)*p['v1']
    v2 = F.get('v2', 1.0)*p['v2']
    v3 = F.get('v3', 1.0)*p['v3']
    v4 = F.get('v4', 1.0)*p['v4']
    w1 = F.get('w1', 1.0)*p['w1']
    w2 = F.get('w2', 1.0)*p['w2']
    d1 = F.get('d1', 1.0)*p['d1']
    d2 = F.get('d2', 1.0)*p['d2']
    d3 = F.get('d3', 1.0)*p['d3']
    vso1 = F.get('vso1', 1.0)*p['vso1']
    vso2 = F.get('vso2', 1.0)*p['vso2']
    wso1 = F.get('wso1', 1.0)*p['wso1']
    wso2 = F.get('wso2', 1.0)*p['wso2']
    return OpticalPotential(
        vv=v1*(1.0 - v2*f + v3*f**2 - v4*f**3),
        wv=w1*f**2/(f**2 + w2**2),
        wd=d1*f**2/(f**2 + d3**2)*np.exp(-d2*f),
        vso=vso1*np.exp(-vso2*f),
        wso=wso1*f**2/(f**2 + wso2**2),
        rv=F.get('rv', 1.0)*p['rv']*a13, av=F.get('av', 1.0)*p['av'],
        rw=F.get('rw', 1.0)*p['rv']*a13, aw=F.get('aw', 1.0)*p['av'],
        rd=F.get('rwd', 1.0)*p['rd']*a13, ad=F.get('awd', 1.0)*p['ad'],
        rso=F.get('rvso', 1.0)*p['rso']*a13, aso=F.get('avso', 1.0)*p['aso'],
        rwso=F.get('rwso', 1.0)*p['rso']*a13,
        awso=F.get('awso', 1.0)*p['aso'],
    )


def koning_delaroche(Z, A, energy, parameters=KD03):
    """Koning-Delaroche global neutron optical potential.

    Parameters
    ----------
    Z : int
        Atomic number of the target
    A : int
        Mass number of the target
    energy : float
        Incident neutron energy in the laboratory system in MeV
    parameters : dict
        Global parameters, e.g. :data:`KD03`

    Returns
    -------
    OpticalPotential
        Potential depths and geometry

    """
    return potential(global_parameters(Z, A, parameters), A, energy)


_TALYS_DATA = None


def _talys_data():
    global _TALYS_DATA
    if _TALYS_DATA is None:
        path = Path(__file__).with_name('talys_neutron_omp.json')
        _TALYS_DATA = json.loads(path.read_text())
    return _TALYS_DATA


def talys_neutron_potential(Z, A, energy):
    """Neutron optical potential as TALYS uses it for TENDL.

    The nucleus-specific Koning-Delaroche parameters distributed with TALYS
    are used when they exist, the global ones otherwise, and the neutron
    optical-model adjustments of the TALYS "best" files are applied. TALYS
    reduces the surface absorption by 15% for nuclei it treats with coupled
    channels; that reduction is compensated by the explicitly coupled levels
    and is therefore not applied in this spherical calculation.

    Parameters
    ----------
    Z : int
        Atomic number of the target
    A : int
        Mass number of the target
    energy : float
        Incident neutron energy in the laboratory system in MeV

    Returns
    -------
    OpticalPotential
        Potential depths and geometry

    """
    data = _talys_data()
    name = f'{ATOMIC_SYMBOL[Z]}{A}'
    params = data['local'].get(name)
    if params is None:
        params = global_parameters(Z, A)
    factors = {key: adjustment_factor(adj, energy)
               for key, adj in data['adjust'].get(name, {}).items()}
    return potential(params, A, energy, factors)


def relativistic_kinematics(energy, target_mass, projectile_mass=NEUTRON_MASS_MEV):
    """Center-of-mass wave number and relativistic reduced mass.

    Uses the prescription of A. Ingemarsson, Phys. Scr. 9, 156 (1974), which
    is what the Koning-Delaroche parameters were fitted with.

    Parameters
    ----------
    energy : float
        Laboratory kinetic energy of the projectile in MeV
    target_mass : float
        Target mass in MeV/c^2
    projectile_mass : float
        Projectile mass in MeV/c^2

    Returns
    -------
    k : float
        Center-of-mass wave number in 1/fm
    mu : float
        Reduced mass in MeV/c^2 to use in the Schrödinger equation

    """
    m, M = projectile_mass, target_mass
    pc = M*np.sqrt(energy*(energy + 2.0*m)/((m + M)**2 + 2.0*M*energy))
    e_p = M*energy/(M + m) + m
    return pc/HBARC, pc**2*e_p/(e_p**2 - m**2)


@dataclass
class ScatteringResult:
    """S-matrix elements and cross sections from an optical-model calculation

    Cross sections are in barns. ``s_plus[l]`` and ``s_minus[l]`` are the
    S-matrix elements for :math:`j = l \\pm 1/2`.

    """
    energy: float
    k: float
    s_plus: np.ndarray
    s_minus: np.ndarray

    @property
    def _l(self):
        return np.arange(self.s_plus.size)

    @property
    def total(self):
        l = self._l
        s = (l + 1)*(1.0 - self.s_plus.real) + l*(1.0 - self.s_minus.real)
        return 2.0*np.pi/self.k**2*s.sum()*0.01

    @property
    def reaction(self):
        l = self._l
        s = ((l + 1)*(1.0 - np.abs(self.s_plus)**2)
             + l*(1.0 - np.abs(self.s_minus)**2))
        return np.pi/self.k**2*s.sum()*0.01

    @property
    def elastic(self):
        return self.total - self.reaction

    def amplitudes(self, mu):
        """Spin-non-flip and spin-flip amplitudes in fm"""
        mu = np.asarray(mu, dtype=float)
        l = self._l
        sp, sm = self.s_plus, self.s_minus
        pl = np.array([lpmv(0, li, mu) for li in l])
        p1 = np.array([lpmv(1, li, mu) for li in l])
        cf = ((l + 1)*(sp - 1.0) + l*(sm - 1.0)) / (2j*self.k)
        cg = (sp - sm) / (2j*self.k)
        return cf @ pl, cg @ p1

    def differential_elastic(self, mu):
        """Center-of-mass elastic differential cross section in b/sr

        Parameters
        ----------
        mu : float or numpy.ndarray
            Cosine of the center-of-mass scattering angle

        """
        f, g = self.amplitudes(mu)
        return (np.abs(f)**2 + np.abs(g)**2)*0.01


def solve(potential, energy, target_mass, step=0.05, lmax=None):
    """Solve the radial Schrödinger equation for a neutron.

    Parameters
    ----------
    potential : OpticalPotential
        Optical potential evaluated at `energy`
    energy : float
        Incident laboratory kinetic energy in MeV
    target_mass : float
        Target mass in MeV/c^2
    step : float
        Radial integration step in fm
    lmax : int, optional
        Highest partial wave. Defaults to :math:`kR + 15`.

    Returns
    -------
    ScatteringResult

    """
    k, mu = relativistic_kinematics(energy, target_mass)
    r_match = potential.radius
    n = int(np.ceil(r_match / step))
    r = step*np.arange(1, n + 1)
    if lmax is None:
        lmax = int(k*r_match) + 15
    l = np.arange(lmax + 1)

    factor = 2.0*mu/HBARC**2
    h12 = step**2/12.0
    S = {}
    for sign in (+1, -1):
        ls = l if sign > 0 else -(l + 1.0)
        # g[i, l] such that u'' = g u
        g = (l*(l + 1.0))[None, :]/r[:, None]**2 - k**2 \
            + factor*potential(r[:, None], ls[None, :])
        w = 1.0 - h12*g
        # Start from the power series of the regular solution near the origin
        q = (l*(l + 1.0))[None, :]/r[:2, None]**2 - g[:2]
        us = np.empty((n, lmax + 1), dtype=complex)
        us[:2] = np.exp((l + 1.0)*np.log(r[:2, None])) \
            * (1.0 - q*r[:2, None]**2/(2.0*(2.0*l + 3.0)))
        u_prev, u = us[0].copy(), us[1].copy()
        for i in range(2, n):
            u_next = ((12.0 - 10.0*w[i-1])*u - w[i-2]*u_prev) / w[i]
            u_prev, u = u, u_next
            big = np.abs(u) > 1e150
            if big.any():
                u_prev[big] /= 1e150
                u[big] /= 1e150
                us[:i, big] /= 1e150
            us[i] = u
        # Two-point matching to Riccati-Hankel functions
        r1, r2 = r[-6], r[-1]
        u1, u2 = us[-6], us[-1]
        hm1, hp1 = _riccati_hankel(l, k*r1)
        hm2, hp2 = _riccati_hankel(l, k*r2)
        S[sign] = (u1*hm2 - u2*hm1)/(u1*hp2 - u2*hp1)
    S[-1][0] = 1.0  # no j = l - 1/2 state for l = 0
    return ScatteringResult(energy, k, S[1], S[-1])


def _riccati_hankel(l, x):
    F = x*spherical_jn(l, x)
    G = -x*spherical_yn(l, x)
    return G - 1j*F, G + 1j*F
