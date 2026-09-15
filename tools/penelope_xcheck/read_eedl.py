"""Read elastic data for one element straight out of EEDL2023.ALL (ENDF-6).

The point is to compare against what survives the ACE route. EEDL gives the
angular distribution as a normalised PDF in mu with LANG=12, i.e. lin-lin
interpolation between the tabulated points. eprdata14 stores the cumulative
distribution instead, and openmc/data/electron.py differentiates that back into
a HISTOGRAM density. Both agree at the nodes; they differ inside every bin.
"""
import re
import numpy as np

def endf_float(s):
    s = s.strip()
    if not s:
        return 0.0
    # ENDF may drop the E: "1.234-5" means 1.234e-5
    m = re.fullmatch(r'([+-]?[\d.]+)([+-]\d+)', s)
    return float(f"{m.group(1)}E{m.group(2)}") if m else float(s)

def fields(line):
    return [endf_float(line[i*11:(i+1)*11]) for i in range(6)]

def ints(line):
    """(C1, C2, L1, L2, N1, N2). Only the last four fields are integers -- the
    first two are floats, so parsing all six as ints trips over records like
    'ZAP=11.0  AWI=5.438673E-4  0  LAW=2  NR=1  NP=2'."""
    out = [endf_float(line[0:11]), endf_float(line[11:22])]
    for i in range(2, 6):
        t = line[i*11:(i+1)*11].strip()
        out.append(int(float(t)) if t else 0)
    return out

def section(path, mat, mf, mt):
    out = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if (len(line) > 75 and line[66:70].strip() and
                    int(line[66:70]) == mat and int(line[70:72]) == mf and
                    int(line[72:75]) == mt):
                out.append(line.rstrip("\n"))
    return out

def read_tab1(lines, k):
    """TAB1 at line k -> (x, y, next_k)."""
    c = ints(lines[k]); nr, np_ = c[4], c[5]; k += 1
    k += (nr*2 + 5)//6            # interpolation ranges, 3 pairs per line
    vals = []
    while len(vals) < 2*np_:
        vals += [v for v in fields(lines[k])]; k += 1
    vals = vals[:2*np_]
    return np.array(vals[0::2]), np.array(vals[1::2]), k

def read_xs(path, mat, mt):
    L = section(path, mat, 23, mt)
    return read_tab1(L, 2)[:2] if False else read_tab1(L, 1)[:2]

def read_angular(path, mat):
    """[(E, mu, pdf, LANG)] for MF=26 MT=525."""
    L = section(path, mat, 26, 525)
    k = 1                                   # skip HEAD
    c = ints(L[k]); law = c[3]; nr, np_ = c[4], c[5]; k += 1
    k += (nr*2 + 5)//6
    k += (2*np_ + 5)//6                     # the yield TAB1
    c = ints(L[k]); nr2, ne = c[4], c[5]; k += 1
    k += (nr2*2 + 5)//6
    out = []
    for _ in range(ne):
        h = ints(L[k]); hv = fields(L[k])
        E = hv[1]; lang = h[2]; nw = h[4]; nl = h[5]; k += 1
        vals = []
        while len(vals) < nw:
            vals += fields(L[k]); k += 1
        vals = vals[:nw]
        out.append((E, np.array(vals[0::2]), np.array(vals[1::2]), lang))
    return out

def moments(mu, f, law):
    """zeroth and <1-mu> under the stated interpolation law."""
    if law == 12:                            # lin-lin in mu
        m0 = np.trapezoid(f, mu)
        m1 = np.trapezoid(f*(1.0-mu), mu)
    else:                                    # histogram
        dx = np.diff(mu)
        m0 = float(np.sum(f[:-1]*dx))
        m1 = float(np.sum(f[:-1]*dx*(1.0-0.5*(mu[1:]+mu[:-1]))))
    return m0, m1
