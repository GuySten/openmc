"""Readers for neutron optical-model parameters distributed with TALYS.

TALYS (A. Koning, S. Hilaire, and S. Goriely, Eur. Phys. J. A 59, 131
(2023)) ships nucleus-specific Koning-Delaroche parameter sets in
``structure/optical/neutron/n-<Element>.omp`` and, for the TENDL library,
per-nuclide parameter adjustments in ``structure/best/<Nuclide>/n-*.talys``.
The functions here read both exactly as TALYS does (``omppar.f90``,
``getvalues.f90``, ``input_omppar.f90``) and evaluate the resulting
adjustment factors as ``adjust.f90`` and ``ompadjust.f90`` do.

"""
from pathlib import Path

#: Optical-model adjustment keywords understood by TALYS ``ompadjust.f90``,
#: without the ``adjust`` suffix
ADJUST_KEYS = (
    'v1', 'v2', 'v3', 'v4', 'rv', 'av', 'w1', 'w2', 'w3', 'w4', 'rw', 'aw',
    'rvd', 'avd', 'd1', 'd2', 'd3', 'rwd', 'awd', 'vso1', 'vso2', 'rvso',
    'avso', 'wso1', 'wso2', 'rwso', 'awso', 'rc'
)


def _fixed(line, widths):
    """Read Fortran F/I edit descriptors from a fixed-format line.

    Blank fields read as zero, as in Fortran.
    """
    values = []
    pos = 0
    for w in widths:
        field = line[pos:pos + w].strip()
        values.append(float(field) if field else 0.0)
        pos += w
    return values


def read_omp_file(path):
    """Read local neutron optical-model parameters from a TALYS .omp file.

    Only the first parameter set of each mass number is returned, which is
    what TALYS uses unless dispersive potentials are requested.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``n-<Element>.omp`` file

    Returns
    -------
    dict
        Mapping of mass number to a dictionary with the Fermi energy ``ef``,
        the radius and diffuseness parameters ``rv``, ``av``, ``rd``, ``ad``,
        ``rso``, ``aso`` (radii in units of :math:`A^{1/3}` fm) and the depth
        parameters ``v1``-``v4``, ``w1``, ``w2``, ``d1``-``d3``, ``vso1``,
        ``vso2``, ``wso1``, ``wso2``. The flag ``coupled`` is set when the
        header marks the set as a coupled-channels potential.

    """
    lines = Path(path).read_text().splitlines()
    sets = {}
    i = 0
    while i < len(lines):
        header = lines[i]
        if not header.strip():
            i += 1
            continue
        # read(2, '(4x, 2i4, 3x, a1)') ia, nomp, omptype
        A = int(header[4:8])
        nomp = int(header[8:12])
        omptype = header[15:16]
        block = lines[i + 1:i + 1 + 4*nomp]
        i += 1 + 4*nomp
        if A in sets:
            continue
        # '(4x, f7.2, f8.3)'
        ef, _ = _fixed(block[0][4:], (7, 8))
        # '(2f8.3, f6.1, f10.4, f9.6, f6.1, f7.1)'
        rv, av, v1, v2, v3, w1, w2 = _fixed(block[1], (8, 8, 6, 10, 9, 6, 7))
        # '(2f8.3, f6.1, f10.4, f7.2)'
        rd, ad, d1, d2, d3 = _fixed(block[2], (8, 8, 6, 10, 7))
        # '(2f8.3, f6.1, f10.4, f6.1, f7.1)'
        rso, aso, vso1, vso2, wso1, wso2 = _fixed(
            block[3], (8, 8, 6, 10, 6, 7))
        sets[A] = {
            'ef': ef, 'rv': rv, 'av': av, 'v1': v1, 'v2': v2, 'v3': v3,
            'v4': 7.0e-9, 'w1': w1, 'w2': w2, 'rd': rd, 'ad': ad, 'd1': d1,
            'd2': d2, 'd3': d3, 'rso': rso, 'aso': aso, 'vso1': vso1,
            'vso2': vso2, 'wso1': wso1, 'wso2': wso2,
            'coupled': omptype.lower() == 'c'
        }
    return sets


def _float(word):
    # Fortran list-directed read of a real number
    return float(word.lower().replace('d', 'e'))


def read_best_adjustments(path, particle='n'):
    """Read optical-model adjustments from a TALYS best file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``<projectile>-<Nuclide>.talys`` file. Tabulated factors
        are looked up next to it and in its ``tables`` subdirectory.
    particle : str
        Particle symbol of the potential to read

    Returns
    -------
    dict
        Mapping of adjustment key (see :data:`ADJUST_KEYS`) to a dictionary
        with the constant factor ``value`` and a list of energy-dependent
        ``entries``, each either ``{'range': [Ea, Eb, Em, D]}`` or
        ``{'table': [[E, factor], ...]}``, in input order.

    Raises
    ------
    ValueError
        If the file contains an optical-model keyword for `particle` that is
        not handled here

    """
    path = Path(path)
    out = {}
    for line in path.read_text().splitlines():
        # TALYS lowercases input lines and splits them on blanks
        words = line.lower().split()
        if len(words) < 3 or words[0].startswith('#'):
            continue
        key = words[0]
        if not key.endswith('adjust') and not key.endswith('adjustf'):
            continue
        if words[1][0] != particle:
            continue
        if key.endswith('adjustf'):
            # Radius and diffuseness bumps of opticalnp.f90 (adjustF)
            raise ValueError(f'Unsupported keyword in {path}: {line}')
        if key[:-6] not in ADJUST_KEYS:
            continue
        name = key[:-6]
        val = _float(words[2])
        # getvalues.f90: tabulated factors when the next word starts with a
        # letter, otherwise four numbers Ea, Eb, Em, D. If any of them is
        # missing the line only sets the constant factor.
        rest = words[3:]
        entry = None
        if rest and rest[0][0].isalpha():
            table = []
            for candidate in (path.parent / rest[0],
                              path.parent / 'tables' / rest[0]):
                if candidate.exists():
                    prev = 0.0
                    for row in candidate.read_text().split('\n'):
                        if not row.split():
                            continue
                        e, d = (_float(x) for x in row.split()[:2])
                        if e == prev:
                            continue
                        if e < prev:
                            raise ValueError(f'Energies must increase in '
                                             f'{candidate}')
                        table.append([e, val*d])
                        prev = e
                    break
            else:
                raise FileNotFoundError(f'{rest[0]} referenced by {path}')
            entry = {'table': table}
        elif len(rest) >= 4:
            entry = {'range': [_float(x) for x in rest[:4]]}
        adj = out.setdefault(name, {'value': 1.0, 'entries': []})
        adj['value'] = val
        if entry is not None:
            adj['entries'].append(entry)
    return out


def adjustment_factor(adjustment, energy):
    """Energy-dependent multiplication factor of one adjusted parameter.

    This reproduces ``adjust.f90`` followed by the multiplication with the
    constant factor in ``ompadjust.f90``. Outside every adjusted energy
    range the factor is the constant one.

    Parameters
    ----------
    adjustment : dict
        One value of the dictionary returned by :func:`read_best_adjustments`
    energy : float
        Incident energy in MeV

    Returns
    -------
    float
        Factor applied to the parameter

    """
    factor = 1.0
    for entry in adjustment['entries']:
        if 'table' in entry:
            table = entry['table']
            if table[0][0] <= energy <= table[-1][0]:
                for (ea, da), (eb, db) in zip(table[:-1], table[1:]):
                    if ea <= energy <= eb:
                        factor = da + (db - da)*(energy - ea)/(eb - ea)
                        break
        else:
            ea, eb, em, d = entry['range']
            if energy <= ea or energy >= eb:
                continue
            npow = 4
            db = (d - 1.0)*(1.0 + 0.5**npow)
            if energy <= em:
                c = ea + 0.5*(em - ea)
                factor = 1.0 + db*(energy - ea)**npow / (
                    (energy - ea)**npow + (c - ea)**npow)
            else:
                c = em + 0.5*(eb - em)
                factor = 1.0 + db*(eb - energy)**npow / (
                    (eb - energy)**npow + (c - eb)**npow)
    return factor*adjustment['value']
